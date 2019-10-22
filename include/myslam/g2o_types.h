/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef MYSLAM_G2O_TYPES_H
#define MYSLAM_G2O_TYPES_H

#include "myslam/common_include.h"
#include "camera.h"

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>


namespace myslam
{
class EdgeProjectXYZRGBD : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    virtual void computeError();
    virtual void linearizeOplus();
    virtual bool read( std::istream& in ){}
    virtual bool write( std::ostream& out) const {}
    
};

// only to optimize the pose, no point ICP
class EdgeProjectXYZRGBDPoseOnly: public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap >
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeProjectXYZRGBDPoseOnly( const Eigen::Vector3d& point ):
        _point(point) {}
    // Error: measure = R*point+t
    virtual void computeError();
    virtual void linearizeOplus();
    
    virtual bool read( std::istream& in ){}
    virtual bool write( std::ostream& out) const {}

protected:
    Eigen::Vector3d _point;
};

class EdgeProjectXYZ2UVPoseOnly: public g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap >
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    virtual void computeError();
    virtual void linearizeOplus();
    
    virtual bool read( std::istream& in ){}
    virtual bool write(std::ostream& os) const {};
    
    Vector3d point_;
    Camera* camera_;
};

class EdgeSE3ProjectDirect: public g2o::BaseUnaryEdge< 1, double, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    /*
    EdgeSE3ProjectDirect() {}

    EdgeSE3ProjectDirect ( Vector3d point, cv::Mat* image )
            : point_ ( point ), image_ ( image )
    {}
    */
    virtual void computeError();
    virtual void linearizeOplus();

    virtual bool read ( std::istream& in ) {}
    virtual bool write ( std::ostream& out ) const {};

    Vector3d point_;   // 3D point in world frame
    Camera* camera_;
    //float cx_=0, cy_=0, fx_=0, fy_=0; // Camera intrinsics
    cv::Mat* image_;    // reference image

protected:
    // get a gray scale value from reference image (bilinear interpolated)双线性插值
    inline float getPixelValue ( float x, float y )
    {
        uchar* data = & image_->data[ int ( y ) * image_->step + int ( x ) ];
        float xx = x - floor ( x );
        float yy = y - floor ( y );
        return float (
                ( 1-xx ) * ( 1-yy ) * data[0] +
                xx* ( 1-yy ) * data[1] +
                ( 1-xx ) *yy*data[ image_->step ] +
                xx*yy*data[image_->step+1]
        );
    }

};

}


#endif // MYSLAM_G2O_TYPES_H
