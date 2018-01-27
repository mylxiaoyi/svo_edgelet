/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include "_modelest.h"
#include <algorithm>
#include <iterator>
#include <limits>
#include <iostream>

using namespace std;

namespace svo {


CvModelEstimator2::CvModelEstimator2(int _modelPoints, CvSize _modelSize, int _maxBasicSolutions)
{
    modelPoints = _modelPoints;
    modelSize = _modelSize;
    maxBasicSolutions = _maxBasicSolutions;
    checkPartialSubsets = true;
    rng = cvRNG(-1);
}

CvModelEstimator2::~CvModelEstimator2()
{
}

void CvModelEstimator2::setSeed( int64 seed )
{
    rng = cvRNG(seed);
}


int CvModelEstimator2::findInliers( const cv::Mat& m1, const cv::Mat& m2,
                                    const cv::Mat& model, cv::Mat& _err,
                                    cv::Mat& _mask, double threshold )
{
//    int i, count = _err->rows*_err->cols, goodCount = 0;
//    const float* err = _err->data.fl;
//    uchar* mask = _mask->data.ptr;
    int i;
    int count = _err.rows*_err.cols;
    int goodCount = 0;
    const float* err = (float*)(_err.data);
    uchar* mask = _mask.data;

    std::cout << "before computeReprojError " << __FILE__ << ":" << __LINE__ << std::endl;
    computeReprojError( m1, m2, model, _err );
    std::cout << "after computeReprojError" << std::endl;
    threshold *= threshold;
    for( i = 0; i < count; i++ )
        goodCount += mask[i] = err[i] <= threshold;
    return goodCount;
}


CV_IMPL int
cvRANSACUpdateNumIters( double p, double ep,
                        int model_points, int max_iters )
{
    if( model_points <= 0 )
        CV_Error( CV_StsOutOfRange, "the number of model points should be positive" );

    p = MAX(p, 0.);
    p = MIN(p, 1.);
    ep = MAX(ep, 0.);
    ep = MIN(ep, 1.);

    // avoid inf's & nan's
    double num = MAX(1. - p, DBL_MIN);
    double denom = 1. - pow(1. - ep,model_points);
    if( denom < DBL_MIN )
        return 0;

    num = log(num);
    denom = log(denom);

    return denom >= 0 || -num >= max_iters*(-denom) ?
        max_iters : cvRound(num/denom);
}

bool CvModelEstimator2::runRANSAC( const cv::Mat& m1, const cv::Mat& m2, cv::Mat& model,
                                    cv::Mat& mask0, double reprojThreshold,
                                    double confidence, int maxIters )
{
    bool result = false;
//    cv::Ptr<CvMat> mask = cvCloneMat(mask0);
//    cv::Ptr<CvMat> models, err, tmask;
//    cv::Ptr<CvMat> ms1, ms2;
    cv::Mat mask = mask0.clone();
    cv::Mat models, err, tmask;
    cv::Mat ms1, ms2;

    int iter, niters = maxIters;
//    int count = m1->rows*m1->cols, maxGoodCount = 0;
    int count = m1.rows*m1.cols, maxGoodCount = 0;
//    CV_Assert( CV_ARE_SIZES_EQ(m1, m2) && CV_ARE_SIZES_EQ(m1, mask) );

    if( count < modelPoints )
        return false;

//    models = cvCreateMat( modelSize.height*maxBasicSolutions, modelSize.width, CV_64FC1 );
//    err = cvCreateMat( 1, count, CV_32FC1 );
//    tmask = cvCreateMat( 1, count, CV_8UC1 );
    std::cout << "modelSize height = " << modelSize.height << ", width = " << modelSize.width << std::endl;
    std::cout << "maxBasicSolutions = " << maxBasicSolutions << std::endl;
    models.create(modelSize.height*maxBasicSolutions, modelSize.width, CV_64FC1 );
    err.create( 1, count, CV_32FC1 );
    tmask.create( 1, count, CV_8UC1 );

    if( count > modelPoints )
    {
        ms1.create( 1, modelPoints, m1.type() );
        ms2.create(1, modelPoints, m2.type() );
    }
    else
    {
        niters = 1;
        ms1 = m1.clone();
        ms2 = m2.clone();
    }

    for( iter = 0; iter < niters; iter++ )
    {
        int i, goodCount, nmodels;
        if( count > modelPoints )
        {
            bool found = getSubset( m1, m2, ms1, ms2, 300 );
            if( !found )
            {
                if( iter == 0 )
                    return false;
                break;
            }
        }

        nmodels = runKernel( ms1, ms2, models );  // find n model
        if( nmodels <= 0 )
            continue;
        std::cout << "models rows = " << models.rows << ", cols = " << models.cols << std::endl;
        std::cout << "nmodels = " << nmodels << std::endl;
        for( i = 0; i < nmodels; i++ )
        {
//            CvMat model_i;
//            cvGetRows( models, &model_i, i*modelSize.height, (i+1)*modelSize.height );
            cv::Mat model_i;
            model_i = models.row(i);
            std::cout << "before findInliers " << __FILE__ << ":" << __LINE__ << std::endl;
            goodCount = findInliers( m1, m2, model_i, err, tmask, reprojThreshold );
            std::cout << "after findINliers" << std::endl;

            if( goodCount > MAX(maxGoodCount, modelPoints-1) )
            {
                std::swap(tmask, mask);
//                cvCopy( &model_i, model );
                model_i.copyTo(model);
                maxGoodCount = goodCount;
                niters = cvRANSACUpdateNumIters( confidence,
                    (double)(count - goodCount)/count, modelPoints, niters );
            }
        }
    }

    if( maxGoodCount > 0 )
    {
//        if( mask != mask0 )
//            cvCopy( mask, mask0 );
        mask.copyTo(mask0);
        result = true;
    }

    return result;
}


//static CV_IMPLEMENT_QSORT( icvSortDistances, int, CV_LT )

bool CvModelEstimator2::runLMeDS( const cv::Mat& m1, const cv::Mat& m2, cv::Mat& model,
                                  cv::Mat& mask, double confidence, int maxIters )
{
    const double outlierRatio = 0.45;
    bool result = false;
//    cv::Ptr<CvMat> models;
//    cv::Ptr<CvMat> ms1, ms2;
//    cv::Ptr<CvMat> err;
    cv::Mat models;
    cv::Mat ms1, ms2;
    cv::Mat err;

    int iter, niters = maxIters;
    int count = m1.rows*m1.cols;
    double minMedian = DBL_MAX, sigma;

//    CV_Assert( CV_ARE_SIZES_EQ(m1, m2) && CV_ARE_SIZES_EQ(m1, mask) );

    if( count < modelPoints )
        return false;

    models.create(modelSize.height*maxBasicSolutions, modelSize.width, CV_64FC1 );
    err.create(1, count, CV_32FC1 );

    if( count > modelPoints )
    {
        ms1.create(1, modelPoints, m1.type());
        ms2.create(1, modelPoints, m2.type());
    }
    else
    {
        niters = 1;
        ms1 = m1.clone();
        ms2 = m2.clone();
    }

    niters = cvRound(log(1-confidence)/log(1-pow(1-outlierRatio,(double)modelPoints)));
    niters = MIN( MAX(niters, 3), maxIters );

    for( iter = 0; iter < niters; iter++ )
    {
        int i, nmodels;
        if( count > modelPoints )
        {
            bool found = getSubset( m1, m2, ms1, ms2, 300 );
            if( !found )
            {
                if( iter == 0 )
                    return false;
                break;
            }
        }

        nmodels = runKernel( ms1, ms2, models );
        if( nmodels <= 0 )
            continue;
        for( i = 0; i < nmodels; i++ )
        {
//            CvMat model_i;
//            cvGetRows( models, &model_i, i*modelSize.height, (i+1)*modelSize.height );
            cv::Mat model_i;
            model_i = models.row(i);
            computeReprojError( m1, m2, model_i, err );
//            icvSortDistances( err->data.i, count, 0 );
            std::sort((int*)(err.data), (int*)(err.data+count));

            double median = count % 2 != 0 ?
                err.data[count/2] : (err.data[count/2-1] + err.data[count/2])*0.5;

            if( median < minMedian )
            {
                minMedian = median;
//                cvCopy( &model_i, model );
                model_i.copyTo(model);
            }
        }
    }

    if( minMedian < DBL_MAX )
    {
        sigma = 2.5*1.4826*(1 + 5./(count - modelPoints))*sqrt(minMedian);
        sigma = MAX( sigma, 0.001 );

        count = findInliers( m1, m2, model, err, mask, sigma );
        result = count >= modelPoints;
    }

    return result;
}


bool CvModelEstimator2::getSubset( const cv::Mat& m1, const cv::Mat& m2,
                                   cv::Mat& ms1, cv::Mat& ms2, int maxAttempts )
{
    cv::AutoBuffer<int> _idx(modelPoints);
    int* idx = _idx;
    int i = 0, j, k, idx_i, iters = 0;
//    int type = CV_MAT_TYPE(m1->type), elemSize = CV_ELEM_SIZE(type);
    int type = m1.type();
    int elemSize = m1.elemSize();
    const int *m1ptr = (int*)(m1.data), *m2ptr = (int*)(m2.data);
    int *ms1ptr = (int*)(ms1.data), *ms2ptr = (int*)(ms2.data);
    int count = m1.cols*m1.rows;

//    assert( CV_IS_MAT_CONT(m1->type & m2->type) && (elemSize % sizeof(int) == 0) );
    elemSize /= sizeof(int);

    for(; iters < maxAttempts; iters++)
    {
        for( i = 0; i < modelPoints && iters < maxAttempts; )
        {
            idx[i] = idx_i = cvRandInt(&rng) % count;
            for( j = 0; j < i; j++ )
                if( idx_i == idx[j] )
                    break;
            if( j < i )
                continue;
            for( k = 0; k < elemSize; k++ )
            {
                ms1ptr[i*elemSize + k] = m1ptr[idx_i*elemSize + k];
                ms2ptr[i*elemSize + k] = m2ptr[idx_i*elemSize + k];
            }
            if( checkPartialSubsets && (!checkSubset( ms1, i+1 ) || !checkSubset( ms2, i+1 )))
            {
                iters++;
                continue;
            }
            i++;
        }
        if( !checkPartialSubsets && i == modelPoints &&
            (!checkSubset( ms1, i ) || !checkSubset( ms2, i )))
            continue;
        break;
    }

    return i == modelPoints && iters < maxAttempts;
}


bool CvModelEstimator2::checkSubset(const cv::Mat &m, int count )
{
    int j, k, i, i0, i1;
    CvPoint2D64f* ptr = (CvPoint2D64f*)(m.data);

    assert( m.type() == CV_64FC2 );

    if( checkPartialSubsets )
        i0 = i1 = count - 1;
    else
        i0 = 0, i1 = count - 1;

    for( i = i0; i <= i1; i++ )
    {
        // check that the i-th selected point does not belong
        // to a line connecting some previously selected points
        for( j = 0; j < i; j++ )
        {
            double dx1 = ptr[j].x - ptr[i].x;
            double dy1 = ptr[j].y - ptr[i].y;
            for( k = 0; k < j; k++ )
            {
                double dx2 = ptr[k].x - ptr[i].x;
                double dy2 = ptr[k].y - ptr[i].y;
                if( fabs(dx2*dy1 - dy2*dx1) <= FLT_EPSILON*(fabs(dx1) + fabs(dy1) + fabs(dx2) + fabs(dy2)))
                    break;
            }
            if( k < j )
                break;
        }
        if( j < i )
            break;
    }

    return i >= i1;
}



class Affine3DEstimator : public CvModelEstimator2
{
public:
    Affine3DEstimator() : CvModelEstimator2(4, cvSize(4, 3), 1) {}
    virtual int runKernel( const cv::Mat& m1, const cv::Mat& m2, cv::Mat& model );
protected:
    virtual void computeReprojError( const cv::Mat& m1, const cv::Mat& m2, const cv::Mat& model, cv::Mat& error );
    virtual bool checkSubset( const cv::Mat& ms1, int count );
};


int Affine3DEstimator::runKernel(const cv::Mat &m1, const cv::Mat &m2, cv::Mat &model )
{
    const cv::Point3d* from = reinterpret_cast<const cv::Point3d*>(m1.data);
    const cv::Point3d* to   = reinterpret_cast<const cv::Point3d*>(m2.data);

    cv::Mat A(12, 12, CV_64F);
    cv::Mat B(12, 1, CV_64F);
    A = cv::Scalar(0.0);

    for(int i = 0; i < modelPoints; ++i)
    {
        *B.ptr<cv::Point3d>(3*i) = to[i];

        double *aptr = A.ptr<double>(3*i);
        for(int k = 0; k < 3; ++k)
        {
            aptr[3] = 1.0;
            *reinterpret_cast<cv::Point3d*>(aptr) = from[i];
            aptr += 16;
        }
    }

    cv::Mat cvA = A;
    cv::Mat cvB = B;
    cv::Mat cvX;
//    cvReshape(model, &cvX, 1, 12);
    cvX = model.reshape(1, 12);
    cvSolve(&cvA, &cvB, &cvX, CV_SVD );

    return 1;
}

void Affine3DEstimator::computeReprojError(const cv::Mat &m1, const cv::Mat &m2, const cv::Mat &model, cv::Mat &error )
{
    int count = m1.rows * m1.cols;
    const cv::Point3d* from = reinterpret_cast<const cv::Point3d*>(m1.data);
    const cv::Point3d* to   = reinterpret_cast<const cv::Point3d*>(m2.data);
    const double* F = (double*)(model.data);
    float* err = (float*)(error.data);

    for(int i = 0; i < count; i++ )
    {
        const cv::Point3d& f = from[i];
        const cv::Point3d& t = to[i];

        double a = F[0]*f.x + F[1]*f.y + F[ 2]*f.z + F[ 3] - t.x;
        double b = F[4]*f.x + F[5]*f.y + F[ 6]*f.z + F[ 7] - t.y;
        double c = F[8]*f.x + F[9]*f.y + F[10]*f.z + F[11] - t.z;

        err[i] = (float)sqrt(a*a + b*b + c*c);
    }
}

bool Affine3DEstimator::checkSubset(const cv::Mat &ms1, int count )
{
    CV_Assert( ms1.type() == CV_64FC3 );

    int j, k, i = count - 1;
    const cv::Point3d* ptr = reinterpret_cast<const cv::Point3d*>(ms1.data);

    // check that the i-th selected point does not belong
    // to a line connecting some previously selected points

    for(j = 0; j < i; ++j)
    {
        cv::Point3d d1 = ptr[j] - ptr[i];
        double n1 = norm(d1);

        for(k = 0; k < j; ++k)
        {
            cv::Point3d d2 = ptr[k] - ptr[i];
            double n = norm(d2) * n1;

            if (fabs(d1.dot(d2) / n) > 0.996)
                break;
        }
        if( k < j )
            break;
    }

    return j == i;
}

int estimateAffine3D(cv::InputArray _from, cv::InputArray _to,
                         cv::OutputArray _out, cv::OutputArray _inliers,
                         double param1, double param2)
{
    cv::Mat from = _from.getMat(), to = _to.getMat();
    int count = from.checkVector(3, CV_32F);

    CV_Assert( count >= 0 && to.checkVector(3, CV_32F) == count );

    _out.create(3, 4, CV_64F);
    cv::Mat out = _out.getMat();

    _inliers.create(count, 1, CV_8U, -1, true);
    cv::Mat inliers = _inliers.getMat();
    inliers = cv::Scalar::all(1);

    cv::Mat dFrom, dTo;
    from.convertTo(dFrom, CV_64F);
    to.convertTo(dTo, CV_64F);

    cv::Mat F3x4 = out;
    cv::Mat mask  = inliers;
    cv::Mat m1 = dFrom;
    cv::Mat m2 = dTo;

    const double epsilon = numeric_limits<double>::epsilon();
    param1 = param1 <= 0 ? 3 : param1;
    param2 = (param2 < epsilon) ? 0.99 : (param2 > 1 - epsilon) ? 0.99 : param2;

    return Affine3DEstimator().runRANSAC(m1, m2, F3x4, mask, param1, param2 );
}

}
