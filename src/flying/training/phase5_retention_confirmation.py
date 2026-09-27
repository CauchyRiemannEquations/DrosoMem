"""Fresh-seed, cross-segment confirmation of anchored prefix weighting."""
import argparse
import sys

from flying.training import phase5_curriculum as engine
from flying.training import phase5_retention as discovery

ARMS = discovery.ARMS
fit_head = discovery.fit_head
OFFSETS = (0, 1000, 2000)


def validate(cfg):
    if cfg.get('experiment') != 'prefix_share_retention_confirmation':
        raise ValueError('Expected prefix_share_retention_confirmation experiment')
    engine.validate({k:v for k,v in cfg.items() if k != 'experiment'})
    if cfg['offsets'] != list(OFFSETS):
        raise ValueError('Confirmation requires all three prespecified offsets')
    if set(cfg['seeds']) & {3142, 3143, 3144}:
        raise ValueError('Confirmation seeds must differ from discovery')


def summary(frame, cfg):
    shared = engine.summary(frame, cfg, intervention='anchored', controls=('fixed','curriculum'))
    criteria = {}
    for offset, groups in shared['by_offset'].items():
        real = groups['fly']
        recall = real['conditions'] == 6 and all(
            c['paired_mean_delta'] > 0 and c['wins'] >= 4 for c in real['comparisons'].values())
        retention = (real['early_retention']['anchored']['lost_at_final'] <=
                     real['early_retention']['fixed']['lost_at_final'] and
                     real['completion_counts']['anchored']['32'] >= real['completion_counts']['fixed']['32'])
        later = real['mean_metrics']['anchored']['later_accuracy'] >= real['mean_metrics']['fixed']['later_accuracy']
        criteria[offset] = dict(recall=bool(recall), retention=bool(retention), later_accuracy=bool(later),
                                primary_success=bool(recall and retention),
                                joint_success=bool(recall and retention and later))
    complete = set(criteria) == {str(offset) for offset in OFFSETS}
    return dict(scope='fresh_seed_cross_segment_confirmation', by_offset=shared['by_offset'],
                criteria_by_offset=criteria,
                all_offsets_primary_passed=complete and all(c['primary_success'] for c in criteria.values()),
                all_offsets_joint_passed=complete and all(c['joint_success'] for c in criteria.values()))


def run(config_path, output, resume=False, max_conditions=None):
    return engine.run(config_path, output, resume, max_conditions, design=sys.modules[__name__])


def verify(output):
    return engine.verify(output, design=sys.modules[__name__])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/phase5_retention_confirmation.json')
    parser.add_argument('--output', default='outputs/phase5_retention_confirmation')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--max-conditions', type=int)
    args = parser.parse_args()
    verify(args.output) if args.verify else run(args.config, args.output, args.resume, args.max_conditions)
