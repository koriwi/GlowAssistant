import math


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def probability_cutoff(threshold, real_option_count):
    # Extra irrelevant options lower this cutoff too; weak matches can slip through.
    return threshold / max(1, real_option_count)
