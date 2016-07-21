#include "poisson_solver.hpp"

#include <iostream>
#include <map>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <opencv2/opencv.hpp>

#pragma clang diagnostic push
#pragma ide diagnostic ignored "IncompatibleTypes"
using namespace std;

namespace solver {

template<typename T>
bool PoissonSolver::BuildMatrix(cv::Mat *img_mask_expand1,
                                cv::Mat *img_gradient_expand1,
                                cv::Mat *img_target_expand1,
                                std::map<int, int> *vector_idx_map,
                                Eigen::SparseMatrix<T> *sparsemat_A,
                                Eigen::VectorXd *vec_b)
{
    /**
     * generate matrix A and vector b, for L(f)= div (v) poisson equation with Dirichlet conditions
     * v is the pasted gradient field, hence div(v) can be calculated by laplacian operator on target image
     * L(f) can be represented by matrix multiplication with toeplitz matrix
     * so the problem can be rewritten as Au = b
     * where A is laplacian operator in toeplitz matrix form
     * b is the divergence of pasted gradient field, which is calculated from target image with laplacian operator
     *
     * cv::Mat *img_mask_expand1 >> input, mask
     * cv::Mat *img_gradient_expand1 >> input, pasted gradient
     * cv::Mat *img_target_expand1 >> input, target area, used in dirchlet boundary condition
     * std::map<int, int> *vector_idx_map >> input, mapping of index between vector and gradient image
     * Eigen::SparseMatrix<T> *sparsemat_A >> output, toeplitz matrix for laplacian filter
     * Eigen::VectorXd *vec_b >> output, vector of pasted gradient map with boundary compensation
     */
    int width = img_mask_expand1->cols;
    int height = img_mask_expand1->rows;
    int channels = img_gradient_expand1->channels();

    std::map<int, int> &vec_mp = *vector_idx_map; //alias
    int nonzero = vec_mp.size(); //long or short

    int counter_row_A = 0;
    //fill sparse matrix A as toeplitz matrix for convolution
    sparsemat_A->reserve(5 * nonzero);
    for (int y = 1; y < height - 1; ++y) {
        uchar *p = img_mask_expand1->ptr<uchar>(y) + 1;
        cv::Vec3d *drv = img_gradient_expand1->ptr<cv::Vec3d>(y) + 1; //scan from the column 1 instead column 0
        for (int x = 1; x < width - 1; ++x, ++p, ++drv) {
            if (*p == 0) continue;

            int idx_current_pixel = y * (width * channels) + (x * channels);
            int idx_top = idx_current_pixel - channels * width;
            int idx_left = idx_current_pixel - channels;
            int idx_right = idx_current_pixel + channels;
            int idx_bottom = idx_current_pixel + channels * width;

            // to optimize insertion
            uchar tlrb = 15; // 0b1111
            if (img_mask_expand1->at<uchar>(y - 1, x) == 0) {
                *drv -= img_target_expand1->at<cv::Vec3b>(y - 1, x); // why drv - img_target not drv - img_gradient
                tlrb &= 7; //0b0111
            }
            if (img_mask_expand1->at<uchar>(y, x - 1) == 0) {
                *drv -= img_target_expand1->at<cv::Vec3b>(y, x - 1);
                tlrb &= 11; //0b1011
            }
            if (img_mask_expand1->at<uchar>(y, x + 1) == 0) {
                *drv -= img_target_expand1->at<cv::Vec3b>(y, x + 1);
                tlrb &= 13; //0b1101
            }
            if (img_mask_expand1->at<uchar>(y + 1, x) == 0) {
                *drv -= img_target_expand1->at<cv::Vec3b>(y + 1, x);
                tlrb &= 14; //0b1110
            }
            // fill sparse matrix A
            for (int k = 0; k < channels; ++k) {
                sparsemat_A->startVec(counter_row_A + k);
                if (tlrb & 8) sparsemat_A->insertBack(vec_mp[idx_top + k], counter_row_A + k) = 1.0; // top
                if (tlrb & 4) sparsemat_A->insertBack(vec_mp[idx_left + k], counter_row_A + k) = 1.0; // left
                sparsemat_A->insertBack(vec_mp[idx_current_pixel + k], counter_row_A + k) = -4.0;// center
                if (tlrb & 2) sparsemat_A->insertBack(vec_mp[idx_right + k], counter_row_A + k) = 1.0; // right
                if (tlrb & 1) sparsemat_A->insertBack(vec_mp[idx_bottom + k], counter_row_A + k) = 1.0; // bottom
            }
            // fill vector b
            (*vec_b)(counter_row_A + 0) = cv::saturate_cast<double>((*drv)[0]);
            (*vec_b)(counter_row_A + 1) = cv::saturate_cast<double>((*drv)[1]);
            (*vec_b)(counter_row_A + 2) = cv::saturate_cast<double>((*drv)[2]);
            counter_row_A += channels;
        }
    }
    sparsemat_A->finalize();
    CV_Assert(nonzero == counter_row_A);

    return true;
}


template<typename T>
bool PoissonSolver::SolvePoissonFunction(const Eigen::SparseMatrix<T> &sparsemat_A,
                                         const Eigen::VectorXd &vec_b,
                                         Eigen::VectorXd *vec_u)
{
    /**
     * more solvers can be choosen, refer to eigen library instruction
     */
    // Solve the (symmetric) system
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> sparseSolver(sparsemat_A);
    *vec_u = sparseSolver.solve(vec_b);
    if(sparseSolver.info() != Eigen::Success)
    {
        throw std::runtime_error("Eigen::SimplicialLDLT Decomposition failed!");
    }
    std::cout << "Eigen::SimplicialLDLT solver succeed" << std::endl;

    return true;
}

bool PoissonSolver::Vec2Cvmat(const Eigen::VectorXd &u,
                              const std::map<int, int> &vector_index_map,
                              cv::Mat *mask,
                              cv::Mat *img_target)
{
    int width = img_target->cols;
    int height = img_target->rows;
    int channels = img_target->channels();
    // copy results back
    int idx = 0;
    for (int y = 0; y < height; ++y) {
        uchar *pd = img_target->ptr<uchar>(y);
        uchar *pm = mask->ptr<uchar>(y);
        for (int x = 0; x < width; ++x, ++pm) {
            if (*pm == 0) {
                pd += 3;
            } else {
                idx = vector_index_map.at(y * (width * channels) + (x * channels)); //because of const, not allowed []
                *pd++ = cv::saturate_cast<uchar>(u[idx + 0]);
                *pd++ = cv::saturate_cast<uchar>(u[idx + 1]);
                *pd++ = cv::saturate_cast<uchar>(u[idx + 2]);
            }
        }
    }

    return true;
}

cv::Mat PoissonSolver::ApplyPoissonFusion(cv::Mat img_src,
                                          cv::Mat img_mask,
                                          cv::Mat img_target,
                                          int offset_x_src2dst,
                                          int offset_y_src2dst)
{
    int offx = offset_x_src2dst;
    int offy = offset_y_src2dst;
    cv::Point offset(offx, offy);

    // calc bounding box
    cv::Point tl(img_mask.size()), br(-1,-1);
    for (int y = 0; y < img_mask.rows; ++y) {
        uchar *p = img_mask.ptr(y);
        for (int x = 0; x < img_mask.cols; ++x, ++p) {
            if (*p == 0) continue;
            if (x < tl.x) tl.x = x;
            if (y < tl.y) tl.y = y;
            if (x > br.x) br.x = x;
            if (y > br.y) br.y = y;
        }
    }
    //compensate boundary for loop
    br.x += 1;
    br.y += 1;

    //add border
    cv::Mat img_src_expand1, img_target_expand1, img_mask_expand1, img_dst_expand1;
    cv::copyMakeBorder(img_src, img_src_expand1, 1, 1, 1, 1, cv::BORDER_REPLICATE); //expend with border pixels
    cv::copyMakeBorder(img_target, img_target_expand1, 1, 1, 1, 1, cv::BORDER_REPLICATE); //expend with border pixels
    cv::copyMakeBorder(img_mask, img_mask_expand1, 1, 1, 1, 1, cv::BORDER_REPLICATE); //expend with border pixels
    img_dst_expand1 = img_target_expand1.clone();

    //extract ROI bounding box
    cv::Rect range_mask_roi(tl, br);
    cv::Rect range_mask_roi_expand1(tl - cv::Point(1, 1), br + cv::Point(1, 1)); //expend 1 pixels

    cv::Mat img_src_roi_expand1 = cv::Mat(img_src_expand1, range_mask_roi_expand1 - cv::Point(1, 1));
    cv::Mat img_target_roi_expand1 = cv::Mat(img_target_expand1, range_mask_roi_expand1 + offset - cv::Point(1, 1));
    cv::Mat img_mask_roi_expand1 = cv::Mat(img_mask, range_mask_roi_expand1);

    //calculate laplacian
    cv::Mat img_src_roi_expand1_64f;
    img_src_roi_expand1.convertTo(img_src_roi_expand1_64f, CV_64F);
    cv::Mat kernel = (cv::Mat_<double>(3,3) << 0, 1,0, 1, -4, 1, 0, 1, 0);

    cv::Mat img_src_roi_expand1_laplacian_64f;
    cv::filter2D(img_src_roi_expand1_64f, img_src_roi_expand1_laplacian_64f, -1, kernel);

    //count nonzero elements, and build index mapping
    int width_roi = range_mask_roi_expand1.width;
    int height_roi = range_mask_roi_expand1.height;
    int channels = img_src.channels();
    int nonzero = 0;

    std::map<int, int> mapping_imgidx_vecidx;
    for (int y = 0; y < height_roi-1; y++) { //loop from 1 to height-1
        uchar *p = img_mask_roi_expand1.ptr(y);

        for (int x = 0; x < width_roi-1; x++, p++) {
            if (*p == 0) continue;

            int id = y * (width_roi * channels) + (x * channels);
            mapping_imgidx_vecidx[id] = nonzero++;   // b
            mapping_imgidx_vecidx[++id] = nonzero++; // g
            mapping_imgidx_vecidx[++id] = nonzero++; // r
        }
    }

    //allocate linear equation, Ax=b
    Eigen::SparseMatrix<double> A = Eigen::SparseMatrix<double>(nonzero, nonzero);
    Eigen::VectorXd b = Eigen::VectorXd(nonzero);
    Eigen::VectorXd u = Eigen::VectorXd(nonzero);

    BuildMatrix(&img_mask_roi_expand1, &img_src_roi_expand1_laplacian_64f, &img_target_roi_expand1 ,&mapping_imgidx_vecidx, &A, &b);
    SolvePoissonFunction(A, b, &u);
    Vec2Cvmat(u, mapping_imgidx_vecidx, &img_mask_roi_expand1, &img_target_roi_expand1);

    cv::Mat img_fusion = img_target_expand1(cv::Rect(1,1,img_target_expand1.cols-1,img_target_expand1.rows-1)).clone();

    return img_fusion;
}

}
#pragma clang diagnostic pop