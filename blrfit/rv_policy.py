"""Explicit reporting policy for development RV measurements.

The corrected shift estimator requires a new on-sky calibration. Legacy
floors are retained as references, never silently applied to the new estimator.
"""
import numpy as np

from . import rv


def measurement_policy(pair, line, kind1, kind2):
    kinds = (kind1, kind2)
    if kinds == ('desi', 'desi'):
        legacy_floor = rv.systematic_floor(line, pair.get('snr_proxy', np.nan))
        legacy_kind = 'desi_sys'
    elif set(kinds) == {'desi', 'sdss'}:
        legacy_floor = rv.cross_survey_floor(line, pair['profile_grade'])
        legacy_kind = 'cross_survey_null_' + pair['profile_grade']
    else:
        legacy_floor, legacy_kind = np.nan, 'unsupported_instrument_pair'
    # One zero-point rule for the whole package (rv.frame_check): the narrow-line
    # frame of the pair is checked once; Hbeta is corrected when the frame is
    # good, Halpha never; a vetoed frame is not a measurement for either line.
    frame_ok = bool(pair.get('frame_ok', False))
    zp_applied = bool(pair.get('zp_applied', False))
    shift = float(pair.get('dv_corrected', pair['dv']))
    stat = float(pair.get('err_corrected', pair['err']))
    valid = bool(np.isfinite(shift) and np.isfinite(stat) and stat > 0
                 and not pair.get('at_bound', False))
    # Cov(broad, narrow) is not available from the present estimator: the
    # zero-point error enters in quadrature, an explicitly labelled
    # approximation, not a calibrated total uncertainty.
    err = stat if valid else np.nan
    per_direction = {k: pair[k] for k in (
        'err_method', 'err_method_ab', 'err_method_ba', 'err_ab', 'err_ba',
        'scale_ab', 'scale_ba', 'chi2_red_ab', 'chi2_red_ba', 'profile_z_ab', 'profile_z_ba',
        'npix_ab', 'npix_ba', 'n_masked_a', 'n_masked_b', 'common_frac', 'search_reach', 'scale_ok', 'ambiguous',
        'dv_alt_ab', 'dv_alt_ba') if k in pair}
    return dict(dv_corrected=shift, zp_applied=zp_applied, err_total=err,
                frame_ok=frame_ok, frame_reason=pair.get('frame_reason', ''),
                zp_source=pair.get('zp_source'),
                statistical_valid=valid, calibration_supported=False,
                uncertainty_policy='statistical_only; zero broad/narrow covariance assumed',
                calibration_status='pending corrected-estimator calibration',
                error_floor=np.nan, error_floor_kind='not_calibrated',
                legacy_error_floor=legacy_floor, legacy_error_floor_kind=legacy_kind,
                profile_stable=pair.get('profile_grade') == 'stable',
                diagnostic_quality_pass=bool(rv.is_reliable(pair)), reliable=False,
                significance=float(abs(shift) / err) if valid and err > 0 else np.nan,
                significance_kind='uncalibrated statistical diagnostic', **per_direction)


def two_line_policy(ha, hb):
    """Use the displayed shifts/errors, with the covariance assumption visible."""
    if not ha.get('measured') or not hb.get('measured'):
        return dict(consistent=False, reason='one line has no usable shift')
    values = [ha.get('dv_corrected', np.nan), hb.get('dv_corrected', np.nan),
              ha.get('err_total', np.nan), hb.get('err_total', np.nan)]
    if not np.all(np.isfinite(values)) or min(values[2:]) <= 0:
        return dict(consistent=False, reason='nonfinite shift or nonpositive uncertainty')
    a, b, ea, eb = values
    result = rv.two_line_consistent(dict(dv=a, err=ea), dict(dv=b, err=eb))
    if result is None:
        return dict(consistent=False, reason='no usable two-line comparison')
    result.update(measurement_policy='displayed corrected shifts and statistical errors',
                  covariance_assumption='zero inter-line covariance; diagnostic only',
                  calibration_supported=False)
    return result
