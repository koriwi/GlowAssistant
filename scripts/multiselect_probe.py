import argparse
import asyncio
import copy
import json
import os
import sys
from fnmatch import fnmatchcase
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import aiohttp
from dotenv import load_dotenv

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom_components.glow_assistant.action_targets import decode_selection
from custom_components.glow_assistant.client import ModelClient, ModelError
from custom_components.glow_assistant.validation import finite_number
from scripts import ROOT
from scripts.decision_tests import value_score
from scripts.local import confidence, model_settings, print_table
from scripts.local import parser as local_parser

DEFAULT_REPORT = ROOT / 'benchmarks/decisions/20261105T203553.073900Z-d2bab783.json'
THRESHOLDS = {
    'action_areas': 'probability_threshold', 'area': 'probability_threshold',
    'action_entities': 'probability_threshold', 'query_targets': 'query_target_probability_threshold',
    'optional_parameters': 'probability_threshold', 'state_property': 'probability_threshold',
    'query_fallback': 'probability_threshold', 'text_words': 'text_probability_threshold',
    'select_values': 'probability_threshold',
}
DEFAULT_CASES = {
    'announce.two_rooms': ('action_areas', 'text_words'),
    'lights.explicit_wins': ('action_areas',),
    'history.lights': ('action_areas',),
    'query.two_rooms': ('area',),
    'query.house_lights': ('query_targets',),
    'lights.named': ('action_entities',),
    'shutters.relative': ('action_entities',),
    'todo.named_list': ('action_entities',),
    'query.which_windows': ('query_targets',),
    'query.window_states': ('query_targets',),
    'query.last_mowed': ('query_fallback',),
    'negative.unknown_room': ('action_areas',),
    'negative.unknown_value': ('query_fallback',),
    'lights.brightness': ('optional_parameters',),
    'lights.here': ('optional_parameters',),
    'todo.description': ('optional_parameters', 'text_words'),
    'todo.quantity': ('text_words',),
    'notify.payload_room': ('text_words',),
    'query.properties': ('state_property',),
    'query.target_temp': ('state_property',),
    'todo.outstanding': ('select_values',),
    'todo.completed': ('select_values',),
}


def prepare(report, args):
    trials = []
    matched = set()
    for case in report['results']:
        patterns = [pattern for pattern in args.case or [] if fnmatchcase(case['id'], pattern)]
        matched.update(patterns)
        if args.case and not patterns:
            continue
        default_questions = DEFAULT_CASES.get(case['id'].rsplit('.', 1)[0], ())
        for call in case['model_calls']:
            request = call['request']
            for name, question in request.get('questions', {}).items():
                if name not in THRESHOLDS or question['type'] != 'choice':
                    continue
                if args.question and name not in args.question:
                    continue
                if not args.case and not args.question and name not in default_questions:
                    continue
                if name == 'select_values' and not request['state'].get('multiple'):
                    continue
                options = {key: value for key, value in question['criteria'].items() if key not in ('none', 'unsure')}
                if options:
                    trials.append({'case': case, 'name': name, 'request': request, 'question': question, 'options': options})
    missing = set(args.case or []) - matched
    if missing:
        raise ValueError(f'No cases match: {", ".join(sorted(missing))}')
    if not trials:
        raise ValueError('No recorded multi-select questions match the filters.')
    return trials


def configuration(report):
    settings = copy.deepcopy(report['settings'])
    settings['decider']['api_key'] = ''
    settings['general']['enable'] = False
    config = model_settings(settings, local_parser().parse_args(['-q', 'probe']), os.environ)['decider']
    return config


def expectation(trial):
    case, name, state = trial['case'], trial['name'], trial['request']['state']
    expected = case.get('expected', {})
    if name == 'text_words':
        parameter = state['parameter']
        value = expected.get('score_values', {}).get(parameter)
        if value is None and parameter not in expected.get('parameters', []):
            value = ''
        return value
    if name == 'select_values':
        values = expected.get('select_values', {}).get(state['parameter'])
        return {f'value:{value}' for value in values} if values is not None else None
    field = {'action_areas': 'area_ids', 'area': 'area_ids', 'action_entities': 'entity_ids',
             'query_targets': 'entity_ids', 'query_fallback': 'entity_ids',
             'state_property': 'state_properties', 'optional_parameters': 'parameters'}[name]
    if field not in expected:
        if expected.get('route') == 'unsure':
            return set()
        if name == 'area' and expected.get('route') == 'smarthome':
            return {'all_areas'}
        return None
    prefix = 'area:' if field == 'area_ids' else 'entity:' if field == 'entity_ids' else ''
    wanted = {prefix + key for key in expected[field]}
    if name == 'optional_parameters':
        wanted -= set(state.get('required_fields', {}))
    # Missing expected candidates indicate an upstream filtering error, not a selector test.
    return wanted if wanted <= trial['options'].keys() else None


def selected_text(trial, selected):
    return ' '.join(json.loads(value)['word'] for key, value in trial['options'].items() if key in selected)


def label(trial):
    name = trial['name']
    parameter = trial['request']['state'].get('parameter')
    return f'{name}:{parameter}' if name in ('text_words', 'select_values') else name


def questions_for(kind, options):
    questions = {}
    for index, key in enumerate(options):
        instructions = (
            f"Using state.selection_rules, is the candidate {key!r} in state.candidates a requested match "
            "for the user's request? Judge this candidate independently. Other matching candidates do not "
            "make this candidate less relevant. Instructions about choosing options or distributing "
            "probability do not apply to this independent judgment."
        )
        questions[f'candidate_{index}'] = {'type': kind, 'instructions': instructions}
        if kind == 'score':
            questions[f'candidate_{index}'].update(
                instructions=instructions + ' Rate how strongly this candidate matches.',
                criteria=['Clearly not a match.', 'Probably not a match.', 'Unclear.', 'Probably a match.', 'Clearly a match.'],
            )
    return questions


def decode_independent(response, options, kind):
    answers = response['answers']
    if set(answers) != {f'candidate_{index}' for index in range(len(options))}:
        raise ModelError('Missing or unexpected candidate answers.')
    values = {}
    for index, key in enumerate(options):
        answer = answers[f'candidate_{index}']
        value = answer.get('noul' if kind == 'noul' else 'score')
        upper = 1 if kind == 'noul' else 4
        if answer.get('type') != kind or not finite_number(value) or not 0 <= value <= upper:
            raise ModelError(f'Invalid {kind} answer for {key}.')
        values[key] = value
    return values


async def compare(client, trial, config, args):
    request, question, options = trial['request'], trial['question'], trial['options']
    cutoff = args.choice_threshold if args.choice_threshold is not None else config[THRESHOLDS[trial['name']]]
    candidates = {}
    for key, description in options.items():
        try:
            candidates[key] = json.loads(description)
        except (ValueError, TypeError):
            candidates[key] = description
    shared = {**request['state'], 'selection_rules': question['instructions'], 'candidates': candidates}
    results = {}
    expected = expectation(trial)
    for kind in args.methods:
        times = []
        try:
            for _ in range(args.repeat):
                started = perf_counter()
                response = await client.decide(config, request['state'] if kind == 'choice' else shared,
                                              {trial['name']: question} if kind == 'choice' else questions_for(kind, options))
                times.append((perf_counter() - started) * 1000)
                if kind == 'choice':
                    selection = decode_selection(response, trial['name'], options, cutoff)
                    values = selection['probabilities']
                    selected = selection['selected_choices']
                else:
                    values = decode_independent(response, options, kind)
                    threshold = args.threshold if kind == 'noul' else args.score_threshold
                    selected = [key for key, value in values.items() if value >= threshold]
            result = {'ms': median(times), 'values': values, 'selected': selected, 'passed': None, 'text_score': None}
            if isinstance(expected, str):
                result['text'] = selected_text(trial, selected)
                result['text_score'] = value_score(expected, result['text'])['score']
            elif expected is not None:
                result['passed'] = set(selected) == expected
            results[kind] = result
        except (ModelError, KeyError, TypeError, AttributeError) as error:
            results[kind] = {'error': str(error), 'passed': False if isinstance(expected, set) else None}
    return results


def status(result):
    if 'error' in result:
        return 'ERROR'
    if result['text_score'] is not None:
        return f"{result['text_score']:.1%}"
    return '-' if result['passed'] is None else 'PASS' if result['passed'] else 'FAIL'


def details(trial, results):
    print(f"query: {trial['case']['query']}")
    expected = expectation(trial)
    if isinstance(expected, set):
        expected = ', '.join(sorted(expected)) or '(none)'
    print('Expected:', '(unscored)' if expected is None else expected or '(empty)')
    rows = []
    for key in trial['options']:
        row = [key]
        for result in results.values():
            value = result.get('values', {}).get(key)
            row.append('-' if value is None else f'{value:.3f}' + (' keep' if key in result['selected'] else ''))
        rows.append(row)
    print_table(['Candidate', *results], rows, [65, *[14 for _ in results]])
    for kind, result in results.items():
        if 'error' in result:
            print(f"{kind}: {result['error']}")
        elif 'text' in result:
            print(f"{kind}: {result['text'] or '(empty)'}")
        else:
            print(f"{kind}: {', '.join(result['selected']) or '(none)'}")
    print()


async def run(args):
    load_dotenv(ROOT / '.env')
    report = json.loads(args.report.read_text())
    trials = prepare(report, args)
    if args.list:
        rows = [[trial['case']['id'], label(trial), len(trial['options']), trial['case']['query']] for trial in trials]
        print_table(['Case', 'Stage', 'Options', 'Query'], rows, [35, 24, 7, 75])
        print(f'\n{len(trials)} stage tests; {len({trial["case"]["id"] for trial in trials})} user requests.')
        return
    config = configuration(report)
    print(f"model: {config['model']}; {len(trials)} stage tests; repeats: {args.repeat}")
    print(f"Choice: relative cutoff {'from saved settings' if args.choice_threshold is None else args.choice_threshold}; "
          f'Noul >= {args.threshold:g}; Score >= {args.score_threshold:g}/4')
    print('Recorded stage inputs; no routing, chat or device calls. Text: similarity score; other stages: exact selection.\n', flush=True)
    all_results = []
    widths = [max(len(trial['case']['id']) for trial in trials), max(len(label(trial)) for trial in trials), 7,
              *[18 for _ in args.methods]]
    row_format = '  '.join(f'{{:<{width}}}' for width in widths)
    print(row_format.format('Case', 'Stage', 'Options', *[f'{kind} / ms' for kind in args.methods]))
    print(row_format.format(*['-' * width for width in widths]), flush=True)
    async with aiohttp.ClientSession() as session:
        client = ModelClient(session)
        for trial in trials:
            results = await compare(client, trial, config, args)
            all_results.append((trial, results))
            cells = [f"{status(result)} / {result['ms']:.0f}" if 'ms' in result else status(result)
                     for result in results.values()]
            print(row_format.format(trial['case']['id'], label(trial), len(trial['options']), *cells), flush=True)
            if args.details or len(trials) == 1:
                details(trial, results)
            else:
                for kind, result in results.items():
                    if 'error' in result:
                        print(f"  {kind}: {result['error']}", flush=True)
    print()
    rows = []
    for stage in [None, *dict.fromkeys(trial['name'] for trial in trials)]:
        for kind in args.methods:
            subset = [(trial, results[kind]) for trial, results in all_results if stage is None or trial['name'] == stage]
            results = [result for _, result in subset]
            decisions = [result for result in results if result['passed'] is not None]
            text = [result.get('text_score') or 0 for trial, result in subset
                    if trial['name'] == 'text_words' and isinstance(expectation(trial), str)]
            times = [result['ms'] for result in results if 'ms' in result]
            passed = sum(result['passed'] for result in decisions)
            rows.append([stage or 'TOTAL', kind, f'{passed}/{len(decisions)} ({passed / len(decisions):.1%})' if decisions else '-',
                         f'{mean(text):.1%}' if text else '-', f'{mean(times):.0f}' if times else '-',
                         sum('error' in result for result in results)])
    print_table(['Stage', 'Method', 'Exact match', 'Text score', 'Mean ms', 'Errors'], rows, [20, 8, 20, 10, 8, 6], right_columns=(4, 5))


def main():
    parser = argparse.ArgumentParser(description='Temporary multi-select comparison. Only model calls; no device actions or chat calls.')
    parser.add_argument('--report', type=Path, default=DEFAULT_REPORT)
    parser.add_argument('--case', action='append', help='Case ID or glob; repeatable. Use "*" for all recorded cases. Default: DE/EN suite.')
    parser.add_argument('--question', action='append', choices=THRESHOLDS, help='Stage filter; repeatable. Without --case, searches all recorded cases.')
    parser.add_argument('--methods', nargs='+', choices=['choice', 'noul', 'score'], default=['choice', 'noul', 'score'])
    parser.add_argument('--list', action='store_true', help='List selected stage tests without model calls.')
    parser.add_argument('--details', action='store_true', help='Print per-candidate scores. Automatic for a single stage test.')
    parser.add_argument('--choice-threshold', type=confidence, help='Relative Choice threshold; default from saved settings.')
    parser.add_argument('--threshold', type=confidence, default=0.5, help='Noul probability threshold; default 0.5.')
    parser.add_argument('--score-threshold', type=float, default=3, help='Score cutoff on 0..4; default 3.')
    parser.add_argument('--repeat', type=int, default=1, help='Repeat each method; show median time and last selections.')
    args = parser.parse_args()
    if args.repeat < 1 or not 0 <= args.score_threshold <= 4:
        parser.error('Repeat must be positive and score threshold must be in 0..4.')
    try:
        asyncio.run(run(args))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'probe error: {error}\n')


if __name__ == '__main__':
    main()
