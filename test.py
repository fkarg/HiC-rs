import warnings
import logging as log
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=PendingDeprecationWarning)

# from hicexplorer import hicCorrectMatrix
from smb.embed import iterative_correct
from hicmatrix import HiCMatrix as hm
from tempfile import NamedTemporaryFile
import os
import numpy as np
import numpy.testing as nt
from matplotlib.testing.compare import compare_images
from hicexplorer.utilities import convertNansToZeros, convertInfsToZeros

from hicexplorer.hicCorrectMatrix import filter_by_zscore, MAD
from scipy.sparse import lil_matrix



ROOT = "/datadisk/Coding/HiCExplorer/hicexplorer/test/test_data/"
# TRAVIS_ROOT = "/datadisk/Coding/HiCExplorer/hicexplorer/test/test_data/"

def main(outfile):
    print('loading matrix and stuff')
    # matrix_filename = ROOT + "small_test_matrix.h5"
    matrix_filename = ROOT + "foo.h5"

    # args.chromosomes
    ma = hm.hiCMatrix(matrix_filename, pChrnameList="chrUextra chr3LHet")
    # ma.reorderChromosomes('chrUextra chr3LHet')

    # mask all zero value bins
    row_sum = np.asarray(ma.matrix.sum(axis=1)).flatten()
    log.info("Removing {} zero value bins".format(sum(row_sum == 0)))
    ma.maskBins(np.flatnonzero(row_sum == 0))
    matrix_shape = ma.matrix.shape

    ma.matrix = convertNansToZeros(ma.matrix)
    ma.matrix = convertInfsToZeros(ma.matrix)


    log.info("matrix contains {} data points. Sparsity {:.3f}.".format(
        len(ma.matrix.data),
        float(len(ma.matrix.data)) / (ma.matrix.shape[0] ** 2)))

    total_filtered_out = set()

    outlier_regions = filter_by_zscore(ma, -1.5, 5.0)
    # compute and print some statistics
    pct_outlier = 100 * float(len(outlier_regions)) / ma.matrix.shape[0]
    ma.printchrtoremove(outlier_regions, label="Bins that are MAD outliers ({:.2f}%) "
                        "out of".format(pct_outlier, ma.matrix.shape[0]),
                        restore_masked_bins=False)

    assert matrix_shape == ma.matrix.shape
    # mask filtered regions
    ma.maskBins(outlier_regions)
    total_filtered_out = set(outlier_regions)

    correction_factors = []
    corrected_matrix = lil_matrix(ma.matrix.shape)

    # corrected_matrix, correction_factors = iterative_correction(ma.matrix)

    # ma.matrix.resize((30, 30))

    print('actually correcting now!')
    correction_factors = iterative_correct(ma.matrix)
    print('done correcting!')

    W = ma.matrix.tocoo()
    W.data *= np.take(correction_factors, W.row)
    W.data *= np.take(correction_factors, W.col)


    corrected_matrix = W.tocsr()

    ma.setMatrixValues(corrected_matrix)
    ma.setCorrectionFactors(correction_factors)

    log.debug("Correction factors {}".format(correction_factors[:10]))

    ma.printchrtoremove(sorted(list(total_filtered_out)),
                        label="Total regions to be removed", restore_masked_bins=False)

    ma.save(outfile.name, pApplyCorrection=False)



def test_correct_matrix_ICE_RUST():
    outfile = NamedTemporaryFile(suffix='.ICE.h5', delete=False)
    outfile.close()

    # args = "correct --matrix {} --correctionMethod ICE --chromosomes "\
    #        "chrUextra chr3LHet --iterNum 500  --outFileName {} "\
    #        "--filterThreshold -1.5 5.0".format(ROOT + "small_test_matrix.h5",
    #                                            outfile.name).split()
    # main(args)
    main(outfile)

    test = hm.hiCMatrix(ROOT + "hicCorrectMatrix/small_test_matrix_ICEcorrected_chrUextra_chr3LHet.h5")
    new = hm.hiCMatrix(outfile.name)
    nt.assert_equal(test.matrix.data, new.matrix.data)
    nt.assert_equal(test.cut_intervals, new.cut_intervals)

    os.unlink(outfile.name)

if __name__ == '__main__':
    test_correct_matrix_ICE_RUST();
