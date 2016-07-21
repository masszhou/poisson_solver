#ifndef __POISSON_SOLVER_H__
#define __POISSON_SOLVER_H__

#include <map>
#include <opencv2/opencv.hpp>
#include <map>
#include <Eigen/Core>
#include <Eigen/Sparse>

//reference information
//developed based on the code from author: idojun
//http://opencv.jp/opencv2-x-samples/poisson-blending

namespace solver {

class PoissonSolver
{
public:
    PoissonSolver() {};
    ~PoissonSolver() {};

    template <typename T>
    bool BuildMatrix(cv::Mat *img_mask_expand1,               //in, mask
                     cv::Mat *img_gradient_expand1,           //in, pasted gradient
                     cv::Mat *img_target_expand1,             //in, dirichlet boundary condition
                     std::map<int, int> *vector_idx_map,      //in, mapping of index between vector and image
                     Eigen::SparseMatrix<T> *sparsemat_A,     //out, toeplitz matrix for laplacian filter
                     Eigen::VectorXd *vec_b);                 //out, vector of pasted gradient with boundary compensation



    template <typename T>
    bool SolvePoissonFunction(const Eigen::SparseMatrix<T> &sparsemat_A,
                              const Eigen::VectorXd &vec_b,
                              Eigen::VectorXd *vec_u);


    bool Vec2Cvmat(const Eigen::VectorXd &u,
                   const std::map<int, int> &vector_index_map,
                   cv::Mat *mask_expand1,
                   cv::Mat *target_img_expand1);

    //integrate application
    cv::Mat ApplyPoissonFusion(cv::Mat src_img,
                               cv::Mat mask,
                               cv::Mat dst_img,
                               int offset_x_src2dst,
                               int offset_y_src2dst);

};

}



#endif /*  __POISSON_SOLVER_H__ */