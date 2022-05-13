import unittest
import os
from fid_helper_pytorch import FidHelper, INPUT_RANGE_PT, INPUT_RANGE_TF


sample_1_dir = os.path.dirname(__file__) + '/sample_1_small'
sample_2_dir = os.path.dirname(__file__) + '/sample_2_small'


class MyTestCase(unittest.TestCase):

    def test_compute_fid_score_from_dir(self):
        fidhelper = FidHelper(':default_1', INPUT_RANGE_TF, device='cuda:0')
        fid = fidhelper.compute_fid_score_from_dir(sample_1_dir, sample_2_dir, 2, 1, verbose=True)
        fid = fidhelper.compute_fid_score_from_dir(sample_1_dir, sample_2_dir, 2, 1, verbose=True)
        fid = fidhelper.compute_fid_score_from_dir(sample_1_dir, sample_2_dir, 2, 1, verbose=True)
        fid = fidhelper.compute_fid_score_from_dir(sample_1_dir, sample_2_dir, 2, 1, verbose=True)
        fid = fidhelper.compute_fid_score_from_dir(sample_1_dir, sample_2_dir, 2, 1, verbose=True)
        fid = fidhelper.compute_fid_score_from_dir(sample_1_dir, sample_2_dir, 2, 1, verbose=True)
        fid = fidhelper.compute_fid_score_from_dir(sample_1_dir, sample_2_dir, 2, 1, verbose=True)
        fid = fidhelper.compute_fid_score_from_dir(sample_1_dir, sample_2_dir, 2, 1, verbose=True)
        self.assertTrue(True)

    def test_load_save_stat_file(self):
        fidhelper = FidHelper(':default_1')
        fidhelper.compute_ref_stat_from_dir(sample_1_dir)
        fidhelper.compute_eval_stat_from_dir(sample_2_dir)
        fid_1 = fidhelper.compute_fid_score()

        fidhelper.save_ref_stat_dict('ref_stat.pkl')
        fidhelper.save_eval_stat_dict('eval_stat.pkl')

        fidhelper = FidHelper(':default_1')
        fidhelper.load_ref_stat_dict('ref_stat.pkl')
        fidhelper.load_eval_stat_dict('eval_stat.pkl')
        fid_2 = fidhelper.compute_fid_score()

        os.unlink('ref_stat.pkl')
        os.unlink('eval_stat.pkl')

        self.assertAlmostEqual(fid_1, fid_2)


if __name__ == '__main__':
    unittest.main()
