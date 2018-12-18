"""
Script entry for evaluation of training & testing results.
Usage:
    1. $ mkdir exp/<DATASET>/<EXPNAME>
    2. Setup configuration in `config.yml`
    3. Setup proto template in `solver.tpl` and `train.tpl`
    4. $ python tools/eval_metrics.py --phase <PHASE> --dataset <DATASET> \
        --expname <EXPNAME> --rsltname <RSLTNAME> --num_workers <NUM_WORKERS>
"""

from _init import workenv, setup

import matplotlib
matplotlib.use('Agg')

import re
import argparse

import numpy as np
import os.path as osp

from terminaltables import AsciiTable
from multiprocessing import Pool, cpu_count

from config import cfg, cfg_from_file
from utils.ops import compute_metrics, may_change_infix
from utils.viz import plot_pr_curve


def compute_metrics_wrapper(clazz, gt, rslt, tiou_thresh):
    ap, (prec, rec) = compute_metrics(gt[clazz], rslt[clazz], tiou_thresh)

    return clazz, ap, prec, rec


def eval_metrics_fixed_nms(phase_key, tiou_thresh, num_workers):
    # Load ground truth and result.
    gt_by_class = np.load({
        'TH14': cfg.DSPEC.TH14_GT,
        'AN': cfg.DSPEC.AN_GT,
    }[cfg.DATASET].format(stage=cfg[phase_key].STAGE))

    rslt_by_class = np.load(cfg[phase_key].RSLT_PATH)

    ap_by_class = np.zeros((cfg.NUM_CLASSES, len(tiou_thresh)))
    # These lists are of shape: [(N_TIOU, N_PRED@TIOU) * N_CLASSES], which
    # cannot be converted into ndarray because the numbers of predictions
    # at different t-IoU thresholds.
    prec_by_class = [[] for _ in range(cfg.NUM_CLASSES)]
    rec_by_class = [[] for _ in range(cfg.NUM_CLASSES)]

    def _append_rslt(rslt):
        ap_by_class[rslt[0]] = rslt[1]
        prec_by_class[rslt[0]] = rslt[2]
        rec_by_class[rslt[0]] = rslt[3]

    num_cpu = cpu_count()
    pool = Pool(min(num_cpu, num_workers))

    jobs = []

    for clazz in range(cfg.NUM_CLASSES):
        # rslt = compute_metrics_wrapper(clazz, gt_by_class,
                                       # rslt_by_class, tiou_thresh)
        # _append_rslt(rslt)
        jobs.append(
            pool.apply_async(
                compute_metrics_wrapper,
                args=(clazz, gt_by_class, rslt_by_class, tiou_thresh),
                callback=_append_rslt
            )
        )

    pool.close()
    pool.join()

    return ap_by_class, prec_by_class, rec_by_class


def eval_metrics(phase_key, tiou_thresh, num_workers):
    if not cfg.EVAL['{}_ADAPTIVE_NMS'.format(phase_key)]:
        return eval_metrics_fixed_nms(phase_key, tiou_thresh, num_workers)
    else:
        def _create_place_holder(m, n):
            return [[[] for _ in range(m)] for _ in range(n)]

        ap_by_class = np.zeros((cfg.NUM_CLASSES, len(tiou_thresh)))
        prec_by_class = _create_place_holder(len(tiou_thresh), cfg.NUM_CLASSES)
        rec_by_class = _create_place_holder(len(tiou_thresh), cfg.NUM_CLASSES)

        for tidx, tiou in enumerate(tiou_thresh):
            # Change {train,test}nms field whenever we are in adaptive mode.
            for pk in ['TEST', 'TRAIN']:
                cfg[pk].NMS_BBOX_THRESH = \
                    tiou - cfg.EVAL['{}_ADAPTIVE_NMS_SHIFT'.format(pk)]

                cfg[phase_key].RSLT_PATH = \
                    may_change_infix(cfg[phase_key].RSLT_PATH, '_',
                                     '{}nms'.format(pk.lower()),
                                     cfg[pk].NMS_BBOX_THRESH)

            cur_ap, cur_prec, cur_rec = eval_metrics_fixed_nms(phase_key,
                                                               tiou_thresh,
                                                               num_workers)

            ap_by_class[:, tidx] = cur_ap[:, tidx]
            for c in range(cfg.NUM_CLASSES):
                prec_by_class[c][tidx] = cur_prec[c][tidx]
                rec_by_class[c][tidx] = cur_rec[c][tidx]

        return ap_by_class, prec_by_class, rec_by_class


def make_table(metric_name, header_name, metric_by_class, header_lst):
    def _fill_table(fld, data, template='{:.03f}', avg=False):
        if isinstance(fld, list) and len(fld) == len(data):
            # Fill multiple rows.
            rows = []
            for idx in range(len(fld)):
                rows += _fill_table(fld[idx], data[idx], template, avg)

            return rows
        else:
            # Fill one row.
            assert len(data.shape) == 1, \
                'Invalid data shape: {}'.format(data.shape)

            row = ['{}'.format(fld)]
            row += map(lambda v: template.format(v), data.tolist())
            if avg:
                row += [template.format(data.mean())]

            return [row]

    title = '{} results on {}'.format(metric_name, cfg.DATASET)

    tiou_row = _fill_table(header_name, tiou_thresh, template='{:.02f}')
    tiou_row[0].append('Avg')
    cls_rows = _fill_table(['{:d}'.format(c) for c in range(cfg.NUM_CLASSES)],
                           metric_by_class, avg=True)

    metric_avg = metric_by_class.mean(axis=0)
    metric_row = _fill_table(metric_name, metric_avg, avg=True)

    table_data = tiou_row + \
        (cls_rows if cfg.EVAL.TBL_INCLUDE_CLS else []) + \
        metric_row

    table = AsciiTable(table_data, title)
    table.justify_columns[-1] = 'right'
    table.inner_footing_row_border = True

    return table


if __name__ == '__main__':
    with workenv():
        old_settings = np.seterr(all='raise', under='ignore')

        parser = argparse.ArgumentParser()

        parser.add_argument('--phase', type=str, required=True,
                            choices=['train', 'test'])
        parser.add_argument('--dataset', type=str, required=True,
                            choices=['TH14', 'AN'])
        parser.add_argument('--expname', type=str, required=True)
        parser.add_argument('--rsltname', type=str, default='rslt')
        parser.add_argument('--num_workers', type=int, default=16)

        args = parser.parse_args()

        config_path = osp.join(cfg.EXP_DIR, args.dataset,
                               args.expname, 'config.yml')

        # Init
        cfg_from_file(config_path)
        args.phase = args.phase.upper()
        setup(args.phase, args.dataset, args.expname, args.rsltname)

        tiou_thresh = np.array(cfg.EVAL.TIOU_THRESH)

        bare_name = osp.splitext(cfg[args.phase].RSLT_PATH)[0]

        if cfg.EVAL['{}_ADAPTIVE_NMS'.format(args.phase)]:
            assert bool(
                re.search('{}nmsada'.format(args.phase.lower()), bare_name)
            ), 'Invalid <rsltname> without {}nmsada'.format(args.phase.lower())

        prc_path = bare_name + '_prc.png'
        map_tbl_path = bare_name + '_map.tbl'
        mpx_tbl_path = bare_name + '_mpx.tbl'

        for pk in ['TEST', 'TRAIN']:
            cfg[args.phase].RSLT_PATH = \
                cfg[args.phase].RSLT_PATH.replace('{}nmsada'.format(pk.lower()),
                                                  '{}nms0.0'.format(pk.lower()))
        print cfg[args.phase].RSLT_PATH

        ap_by_class, prec_by_class, rec_by_class = \
            eval_metrics(args.phase, tiou_thresh, args.num_workers)

        px_by_class = np.zeros((cfg.NUM_CLASSES, len(tiou_thresh)))
        for c in range(cfg.NUM_CLASSES):
            try:
                num_pred = len(prec_by_class[c][0])
                idx = int(np.floor(num_pred * cfg.EVAL.PREC_AT_TOPX)) - 1
                px_by_class[c] = np.array(map(lambda t: prec_by_class[c][t][idx],
                                              range(len(tiou_thresh))))
            except IndexError:
                px_by_class[c, ...] = 0

        if cfg.VIZ.PLOT_PR_CURVE:
            plot_pr_curve(prec_by_class, rec_by_class,
                          tiou_thresh, prc_path)

        map_tbl = make_table('mAP', 't-IoU',
                             ap_by_class, tiou_thresh)
        mpx_tbl = make_table('mP@{:.0%}'.format(cfg.EVAL.PREC_AT_TOPX), 't-IoU',
                             px_by_class, tiou_thresh)

        with open(map_tbl_path, 'w') as fmap, open(mpx_tbl_path, 'w') as fmpx:
            fmap.write(map_tbl.table)
            fmpx.write(mpx_tbl.table)

        print map_tbl.table, '\n', mpx_tbl.table, '\n', \
            'Result saved at {} & {}'.format(map_tbl_path, mpx_tbl_path)
