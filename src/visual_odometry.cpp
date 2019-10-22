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

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>
#include <boost/timer.hpp>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"
#include "myslam/g2o_types.h"

namespace myslam
{

VisualOdometry::VisualOdometry() :
    state_ ( INITIALIZING ), ref_ ( nullptr ), curr_ ( nullptr ), map_ ( new Map ), num_lost_ ( 0 ), num_inliers_ ( 0 ), matcher_flann_ ( new cv::flann::LshIndexParams ( 5,10,2 ) )
{
    num_of_features_    = Config::get<int> ( "number_of_features" );
    scale_factor_       = Config::get<double> ( "scale_factor" );
    level_pyramid_      = Config::get<int> ( "level_pyramid" );
    match_ratio_        = Config::get<float> ( "match_ratio" );
    max_num_lost_       = Config::get<float> ( "max_num_lost" );
    min_inliers_        = Config::get<int> ( "min_inliers" );
    key_frame_min_rot   = Config::get<double> ( "keyframe_rotation" );
    key_frame_min_trans = Config::get<double> ( "keyframe_translation" );
    map_point_erase_ratio_ = Config::get<double> ( "map_point_erase_ratio" );
    orb_ = cv::ORB::create ( num_of_features_, scale_factor_, level_pyramid_ );
    //Tcw = Eigen::Isometry3d::Identity();
}

VisualOdometry::~VisualOdometry()
{

}

bool VisualOdometry::addFrame ( Frame::Ptr frame )
{
    switch ( state_ )
    {
    case INITIALIZING:
    {
        state_ = OK;
        curr_ = ref_ = frame;
        // extract features from first frame and add them into map
        extractKeyPoints();
        computeDescriptors();
        addKeyFrame();      // the first frame is a key-frame
        break;
    }
    case OK:
    {
        curr_ = frame;
        curr_->T_c_w_ = ref_->T_c_w_;
        extractKeyPoints();
        computeDescriptors();
        featureMatching();
        poseEstimationPnP();
        if ( checkEstimatedPose() == true ) // a good estimation
        {
            curr_->T_c_w_ = T_c_w_estimated_;
            optimizeMap();
            num_lost_ = 0;
            if ( checkKeyFrame() == true ) // is a key-frame
            {
                addKeyFrame();
                //ref_ = curr_;

            }
        }
        else // bad estimation due to various reasons
        {
            num_lost_++;
            if ( num_lost_ > max_num_lost_ )
            {
                state_ = LOST;
            }
            return false;
        }

        cout<< "ref: "<<ref_->T_c_w_<<endl;
        break;
    }
    case LOST:
    {
        cout<<"vo has lost."<<endl;
        break;
    }
    }
    return true;
}

bool VisualOdometry::addFrameDirect( Frame::Ptr frame )
{
    switch ( state_ )
    {
        case INITIALIZING:
        {
            state_ = OK;
            curr_ = ref_ = frame;

            //------select the pixels with high gradiants
            for ( int x=10; x<curr_->gray_.cols-10; x++ )
                for ( int y=10; y<curr_->gray_.rows-10; y++ )
                {
                    Vector2d p (x,y);
                    Vector2d delta (
                            curr_->gray_.ptr<uchar>(y)[x+1] - curr_->gray_.ptr<uchar>(y)[x-1],
                            curr_->gray_.ptr<uchar>(y+1)[x] - curr_->gray_.ptr<uchar>(y-1)[x]
                    );
                    if ( delta.norm() < 50 )
                        continue;
                    ushort d = curr_->depth_.ptr<ushort> (y)[x];
                    if ( d==0 )
                        continue;
                    Vector3d p3d = ref_->camera_->pixel2camera( p, d/ref_->camera_->depth_scale_ );
                    float grayscale = float ( curr_->gray_.ptr<uchar> (y) [x] );
                    measurements_.push_back ( Measurement ( p3d, grayscale ) );
                }

            // ------using spare keypoints
            //extractKeyPointsDirect();
//            for ( auto kp : keypoints_curr_ )
//            {
//                // 去掉邻近边缘处的点
//                if ( kp.pt.x < 20 || kp.pt.y < 20 || ( kp.pt.x+20 ) >curr_->color_.cols || ( kp.pt.y+20 ) >curr_->color_.rows )
//                    continue;
//                ushort d = curr_->depth_.ptr<ushort> ( cvRound ( kp.pt.y ) ) [ cvRound ( kp.pt.x ) ];
//                if ( d==0 )
//                    continue;
//                Vector2d p ( kp.pt.x, kp.pt.y );
//                Eigen::Vector3d p3d =ref_->camera_->pixel2camera ( p, d/ref_->camera_->depth_scale_ );
//                float grayscale = float ( curr_->gray_.ptr<uchar> ( cvRound ( kp.pt.y ) ) [ cvRound ( kp.pt.x ) ] );
//                measurements_.push_back ( Measurement ( p3d, grayscale ) );
//            }
            addKeyFrame();      // the first frame is a key-frame
            break;
        }
        case OK:
        {
            curr_ = frame;
            curr_->T_c_w_ = ref_->T_c_w_;
            //featureMatching();
            poseEstimationDirect();

            SE3 T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
            Sophus::Vector6d d = T_r_c.log();
            //cout << "d.norm() = " << d << endl;
            if ( d.norm() <= 5.0 )// a good estimation
            {
                curr_->T_c_w_ = T_c_w_estimated_;
                optimizeMap();
                num_lost_ = 0;
                //cout << checkKeyFrame() << endl;
                if ( checkKeyFrame() == true ) // is a key-frame
                {
                    addKeyFrame();
                }

                //ref_ = curr_;
            }
            else // bad estimation due to various reasons
            {
                num_lost_++;
                if ( num_lost_ > max_num_lost_ )
                {
                    state_ = LOST;
                }
                return false;
            }
            break;
        }
        case LOST:
        {
            cout<<"vo has lost."<<endl;
            break;
        }
    }
    return true;
}

bool VisualOdometry::addFrameDNA( Frame::Ptr frame )
    {
        switch ( state_ )
        {
            case INITIALIZING:
            {
                state_ = OK;
                curr_ = ref_ = frame;

                //------select the pixels with high gradiants
                for ( int x=10; x<curr_->gray_.cols-10; x++ )
                    for ( int y=10; y<curr_->gray_.rows-10; y++ )
                    {
                        Vector2d p (x,y);
                        Vector2d delta (
                                curr_->gray_.ptr<uchar>(y)[x-1] - curr_->gray_.ptr<uchar>(y)[x+1],
                                curr_->gray_.ptr<uchar>(y-1)[x] - curr_->gray_.ptr<uchar>(y+1)[x]
                        );
                        cout << "derivative: "<<delta.norm() <<endl;
                        if ( delta.norm() < 50 )
                            continue;
                        ushort d = curr_->depth_.ptr<ushort> (y)[x];
                        if ( d==0 )
                            continue;
                        Vector3d p3d = ref_->camera_->pixel2camera( p, d/ref_->camera_->depth_scale_ );
                        float grayscale = float ( curr_->gray_.ptr<uchar> (y) [x] );
                        measurements_.push_back ( Measurement ( p3d, grayscale ) );
                    }

                addKeyFrame();      // the first frame is a key-frame
                break;
            }
            case OK:
            {
                curr_ = frame;
                curr_->T_c_w_ = ref_->T_c_w_;
                //featureMatching();
                poseEstimationDNA();

                SE3 T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
                Sophus::Vector6d d = T_r_c.log();
                //cout << "d.norm() = " << d << endl;
                if ( d.norm() <= 5.0 )// a good estimation
                {
                    curr_->T_c_w_ = T_c_w_estimated_;
                    optimizeMap();
                    num_lost_ = 0;
                    //cout << checkKeyFrame() << endl;
                    if ( checkKeyFrame() == true ) // is a key-frame
                    {
                        addKeyFrame();
                    }

                    //ref_ = curr_;
                }
                else // bad estimation due to various reasons
                {
                    num_lost_++;
                    if ( num_lost_ > max_num_lost_ )
                    {
                        state_ = LOST;
                    }
                    return false;
                }
                break;
            }
            case LOST:
            {
                cout<<"vo has lost."<<endl;
                break;
            }
        }
        return true;
    }

void VisualOdometry::extractKeyPoints()
{
    boost::timer timer;
    orb_->detect ( curr_->color_, keypoints_curr_ );
    cout<<"extract keypoints cost time: "<<timer.elapsed() <<endl;
}

void VisualOdometry::extractKeyPointsDirect()
{
    boost::timer timer;
    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
    detector->detect( curr_->color_, keypoints_curr_ );
    cout<<"extract keypoints cost time: "<<timer.elapsed() <<endl;
}

void VisualOdometry::computeDescriptors()
{
    boost::timer timer;
    orb_->compute ( curr_->color_, keypoints_curr_, descriptors_curr_ );
    cout<<"descriptor computation cost time: "<<timer.elapsed() <<endl;
}

void VisualOdometry::featureMatching()
{
    boost::timer timer;
    vector<cv::DMatch> matches;
    // select the candidates in map 
    Mat desp_map;
    vector<MapPoint::Ptr> candidate;
    for ( auto& allpoints: map_->map_points_ )
    {
        MapPoint::Ptr& p = allpoints.second;
        // check if p in curr frame image 
        if ( curr_->isInFrame(p->pos_) )
        {
            // add to candidate 
            p->visible_times_++;
            candidate.push_back( p );
            desp_map.push_back( p->descriptor_ );
        }
    }
    
    matcher_flann_.match ( desp_map, descriptors_curr_, matches );
    // select the best matches
    float min_dis = std::min_element (
                        matches.begin(), matches.end(),
                        [] ( const cv::DMatch& m1, const cv::DMatch& m2 )
    {
        return m1.distance < m2.distance;
    } )->distance;

    match_3dpts_.clear();
    match_2dkp_index_.clear();
    for ( cv::DMatch& m : matches )
    {
        if ( m.distance < max<float> ( min_dis*match_ratio_, 30.0 ) )
        {
            match_3dpts_.push_back( candidate[m.queryIdx] );
            match_2dkp_index_.push_back( m.trainIdx );
        }
    }
    cout<<"good matches: "<<match_3dpts_.size() <<endl;
    cout<<"match cost time: "<<timer.elapsed() <<endl;
}

void VisualOdometry::poseEstimationPnP()
{
    // construct the 3d 2d observations
    vector<cv::Point3f> pts3d;
    vector<cv::Point2f> pts2d;

    for ( int index:match_2dkp_index_ )
    {
        pts2d.push_back ( keypoints_curr_[index].pt );
    }
    for ( MapPoint::Ptr pt:match_3dpts_ )
    {
        pts3d.push_back( pt->getPositionCV() );
    }

    Mat K = ( cv::Mat_<double> ( 3,3 ) <<
              ref_->camera_->fx_, 0, ref_->camera_->cx_,
              0, ref_->camera_->fy_, ref_->camera_->cy_,
              0,0,1
            );
    Mat rvec, tvec, inliers;
    cv::solvePnPRansac ( pts3d, pts2d, K, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers );
    num_inliers_ = inliers.rows;
    cout<<"pnp inliers: "<<num_inliers_<<endl;
    //cout << "rvec= " << rvec << endl;
    T_c_w_estimated_ = SE3 (
                           SO3 ( rvec.at<double> ( 0,0 ), rvec.at<double> ( 1,0 ), rvec.at<double> ( 2,0 ) ),
                           Vector3d ( tvec.at<double> ( 0,0 ), tvec.at<double> ( 1,0 ), tvec.at<double> ( 2,0 ) )
                       );

    // using bundle adjustment to optimize the pose
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,2>> Block;
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    Block* solver_ptr = new Block ( linearSolver );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setId ( 0 );
    pose->setEstimate ( g2o::SE3Quat (
        T_c_w_estimated_.rotation_matrix(), T_c_w_estimated_.translation()
    ));
    optimizer.addVertex ( pose );

    // edges
    for ( int i=0; i<inliers.rows; i++ )
    {
        int index = inliers.at<int> ( i,0 );
        // 3D -> 2D projection
        EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();
        edge->setId ( i );
        edge->setVertex ( 0, pose );
        edge->camera_ = curr_->camera_.get();
        edge->point_ = Vector3d ( pts3d[index].x, pts3d[index].y, pts3d[index].z );
        edge->setMeasurement ( Vector2d ( pts2d[index].x, pts2d[index].y ) );
        edge->setInformation ( Eigen::Matrix2d::Identity() );
        optimizer.addEdge ( edge );
        // set the inlier map points 
        match_3dpts_[index]->matched_times_++;
    }

    optimizer.initializeOptimization();
    optimizer.optimize ( 10 );

    T_c_w_estimated_ = SE3 (
        pose->estimate().rotation(),
        pose->estimate().translation()
    );
    
    cout<<"T_c_w_estimated_: "<<endl<<T_c_w_estimated_.matrix()<<endl;
}

void VisualOdometry::poseEstimationICP() {
    vector<cv::Point3f> pts1;
    vector<cv::Point3f> pts2;

    Mat K = (cv::Mat_<double>(3, 3) <<
                                    ref_->camera_->fx_, 0, ref_->camera_->cx_,
            0, ref_->camera_->fy_, ref_->camera_->cy_,
            0, 0, 1
    );

    for (MapPoint::Ptr pt:match_3dpts_) {
        pts1.push_back(pt->getPositionCV());
        //cout << pt->getPositionCV()  << endl;
    }
    //cout << endl;
    for (int index:match_2dkp_index_) {
        ushort d = curr_->depth_.ptr<ushort>(cvRound(keypoints_curr_[index].pt.y))[cvRound(
                keypoints_curr_[index].pt.x)];
        if (d == 0)
            continue;
        cv::Point2d p(
                (keypoints_curr_[index].pt.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                (keypoints_curr_[index].pt.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
        );
        float dd = float(d) / curr_->camera_->depth_scale_;
        pts2.push_back(cv::Point3f(p.x * dd, p.y * dd, dd));
        //cout << cv::Point3f ( p.x*dd, p.y*dd, dd ) << endl;
    }
    int N = pts1.size();

    // pose estimation in SVD
    cv::Point3f p1, p2;
    for ( int i=0; i<N; i++ )
    {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = cv::Point3f( cv::Vec3f(p1) / N );
    p2 = cv::Point3f( cv::Vec3f(p2) / N );
    vector<cv::Point3f> q1( N ), q2( N );
    for ( int i=0; i<N; i++ )
    {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }
    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for ( int i=0; i<N; i++ )
    {
        W+=Eigen::Vector3d( q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d ( q2[i].x, q2[i].y, q2[i].z ).transpose();
    }
    // SVD
    Eigen::JacobiSVD<Eigen::Matrix3d> svd ( W, Eigen::ComputeFullU|Eigen::ComputeFullV );
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    if (U.determinant() * V.determinant() < 0)
    {
        for (int x=0; x<3; x++)
        {
            U(x, 2) *= -1;
        }
    }

    Eigen::Matrix3d R = U* ( V.transpose() );
    Eigen::Vector3d t = Eigen::Vector3d ( p1.x, p1.y, p1.z ) - R * Eigen::Vector3d ( p2.x, p2.y, p2.z );

    T_c_w_estimated_ = SE3 ( SO3 (R), t );
    num_inliers_ = N;
    cout<<"T_c_w_estimated_: "<<T_c_w_estimated_<<endl;

    // using bundle adjustment to optimize t he pose
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>> Block;
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    Block* solver_ptr = new Block ( linearSolver );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );

    // vertex
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setId ( 0 );
    pose->setEstimate ( g2o::SE3Quat (
            T_c_w_estimated_.rotation_matrix(), T_c_w_estimated_.translation()
    ));
    optimizer.addVertex ( pose );

    //edges
    int index = 1;
    vector<EdgeProjectXYZRGBDPoseOnly*> edges;
    for ( size_t i=0; i<N; i++ )
    {
        EdgeProjectXYZRGBDPoseOnly* edge = new EdgeProjectXYZRGBDPoseOnly(
                Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z) );
        edge->setId( index );
        edge->setVertex( 0, dynamic_cast<g2o::VertexSE3Expmap*> (pose) );
        edge->setMeasurement( Eigen::Vector3d(
                pts1[i].x, pts1[i].y, pts1[i].z) );
        edge->setInformation( Eigen::Matrix3d::Identity()*1e4 );
        optimizer.addEdge(edge);
        index++;
        edges.push_back(edge);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    T_c_w_estimated_ = SE3 (
            pose->estimate().rotation(),
            pose->estimate().translation()
    );

    cout<<"T_c_w_estimated_: "<<endl<<T_c_w_estimated_.matrix()<<endl;

}





bool VisualOdometry::checkEstimatedPose()
{
    // check if the estimated pose is good
    if ( num_inliers_ < min_inliers_ )
    {
        cout<<"reject because inlier is too small: "<<num_inliers_<<endl;
        return false;
    }
    // if the motion is too large, it is probably wrong
    SE3 T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
    Sophus::Vector6d d = T_r_c.log();
    if ( d.norm() > 5.0 )
    {
        cout<<"reject because motion is too large: "<<d.norm() <<endl;
        return false;
    }
    return true;
}

bool VisualOdometry::checkKeyFrame()
{
    SE3 T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
    Sophus::Vector6d d = T_r_c.log();
    Vector3d trans = d.head<3>();
    Vector3d rot = d.tail<3>();
    cout<<"rot="<<rot.norm()<<" trans="<<trans.norm()<<endl;
    if ( rot.norm() >key_frame_min_rot || trans.norm() >key_frame_min_trans )
        return true;
    return false;
}

void VisualOdometry::addKeyFrame()
{
    // Add keypoints descriptors in direct
    if ( descriptors_curr_.empty() )
    {
        computeDescriptors();
    }

    if ( map_->keyframes_.empty() )
    {
        // first key-frame, add all 3d points into map
        for ( size_t i=0; i<keypoints_curr_.size(); i++ )
        {
            double d = curr_->findDepth ( keypoints_curr_[i] );
            if ( d < 0 ) 
                continue;
            Vector3d p_world = ref_->camera_->pixel2world (
                Vector2d ( keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y ), curr_->T_c_w_, d
            );
            Vector3d n = p_world - ref_->getCamCenter();
            n.normalize();
            MapPoint::Ptr map_point = MapPoint::createMapPoint(
                p_world, n, descriptors_curr_.row(i).clone(), curr_.get()
            );
            map_->insertMapPoint( map_point );
        }
    }

    map_->insertKeyFrame ( curr_ );
    ref_ = curr_;
}

void VisualOdometry::addMapPoints()
{
    // add the new map points into map
    vector<bool> matched(keypoints_curr_.size(), false); 
    for ( int index:match_2dkp_index_ )
        matched[index] = true;
    for ( int i=0; i<keypoints_curr_.size(); i++ )
    {
        if ( matched[i] == true )   
            continue;
        double d = ref_->findDepth ( keypoints_curr_[i] );
        if ( d<0 )  
            continue;
        Vector3d p_world = ref_->camera_->pixel2world (
            Vector2d ( keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y ), 
            curr_->T_c_w_, d
        );
        Vector3d n = p_world - ref_->getCamCenter();
        n.normalize();
        MapPoint::Ptr map_point = MapPoint::createMapPoint(
            p_world, n, descriptors_curr_.row(i).clone(), curr_.get()
        );
        map_->insertMapPoint( map_point );
    }
}

void VisualOdometry::optimizeMap()
{
    // remove the hardly seen and no visible points
    for ( auto iter = map_->map_points_.begin(); iter != map_->map_points_.end(); )
    {
        if ( !curr_->isInFrame(iter->second->pos_) )
        {
            iter = map_->map_points_.erase(iter);
            continue;
        }
        float match_ratio = float(iter->second->matched_times_)/iter->second->visible_times_;
        if ( match_ratio < map_point_erase_ratio_ )
        {
            iter = map_->map_points_.erase(iter);
            continue;
        }
        
        double angle = getViewAngle( curr_, iter->second );
        if ( angle > M_PI/6. )
        {
            iter = map_->map_points_.erase(iter);
            continue;
        }
        if ( iter->second->good_ == false )
        {
            // TODO try triangulate this map point 
        }
        iter++;
    }
    
    if ( match_2dkp_index_.size()<100 )
        addMapPoints();
    if ( map_->map_points_.size() > 1000 )  
    {
        // TODO map is too large, remove some one 
        map_point_erase_ratio_ += 0.05;
    }
    else 
        map_point_erase_ratio_ = 0.1;
    cout<<"map points: "<<map_->map_points_.size()<<endl;
}

double VisualOdometry::getViewAngle ( Frame::Ptr frame, MapPoint::Ptr point )
{
    Vector3d n = point->pos_ - frame->getCamCenter();
    n.normalize();
    return acos( n.transpose()*point->norm_ );
}

int VisualOdometry::initializing(Frame::Ptr initframe )
{
    ref_ = curr_ = initframe;
    extractKeyPoints();
    //init_orb_->detect( initframe->color_, init_keypoints_ );
    /*
    cout<<keypoints_curr_.size()<<endl;
    SE3 T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
    Sophus::Vector6d d = T_r_c.log();
    cout << "d.norm() = " << d.norm() << endl;
    if ( d.norm() <= 5.0 )// a good estimation
     */
    if(keypoints_curr_.size() < 300 )
    {
        cout<<"using Direct Method"<<endl;
        return 0;
    }
//    else
//        cout<<"using Feature Method"<<endl;
//        return 1;
    return -1;
}

// 直接法估计位姿
// 输入：测量值（空间点的灰度），新的灰度图，相机内参； 输出：相机位姿
// 返回：true为成功，false失败
void VisualOdometry::poseEstimationDirect ()
{
    //cout<<"T_c_w_estimated_: "<<endl<<T_c_w_estimated_.matrix()<<endl;
    // using bundle adjustment to optimize the pose
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,2>> DirectBlock;  // 求解的向量是6＊1的
    DirectBlock::LinearSolverType* linearSolver = new g2o::LinearSolverDense< DirectBlock::PoseMatrixType > ();
    DirectBlock* solver_ptr = new DirectBlock ( linearSolver );
    // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr ); // G-N
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr ); // L-M
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setId ( 0 );
    pose->setEstimate ( g2o::SE3Quat (
            T_c_w_estimated_.rotation_matrix(), T_c_w_estimated_.translation()
    ));

    optimizer.addVertex ( pose );

    // 添加边
    int index = 1;
    for ( Measurement m: measurements_ )
    {
        EdgeSE3ProjectDirect* edge = new EdgeSE3ProjectDirect ();
        edge->setVertex ( 0, pose );
        edge->camera_ = curr_->camera_.get();
        edge->point_ = m.pos_world;
        edge->image_ = &(curr_->gray_);
        edge->setMeasurement ( m.grayscale );
        edge->setInformation ( Eigen::Matrix<double,1,1>::Identity() );
        edge->setId ( index++ );
        optimizer.addEdge ( edge );
    }
    cout<<"edges in graph: "<<optimizer.edges().size() <<endl;
    optimizer.initializeOptimization();
    optimizer.optimize ( 30 );
    T_c_w_estimated_ = SE3 (
            pose->estimate().rotation(),
            pose->estimate().translation()
    );
    cout<<"T_c_w_estimated_: "<<endl<<T_c_w_estimated_.matrix()<<endl;
}

void VisualOdometry::poseEstimationDNA ()
    {
        //cout<<"T_c_w_estimated_: "<<endl<<T_c_w_estimated_.matrix()<<endl;
        // using bundle adjustment to optimize the pose
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,2>> DirectBlock;  // 求解的向量是6＊1的
        DirectBlock::LinearSolverType* linearSolver = new g2o::LinearSolverDense< DirectBlock::PoseMatrixType > ();
        DirectBlock* solver_ptr = new DirectBlock ( linearSolver );
        // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr ); // G-N
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr ); // L-M
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm ( solver );

        g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
        pose->setId ( 0 );
        pose->setEstimate ( g2o::SE3Quat (
                T_c_w_estimated_.rotation_matrix(), T_c_w_estimated_.translation()
        ));

        optimizer.addVertex ( pose );

        // 添加边
        int index = 1;
        for ( Measurement m: measurements_ )
        {
            EdgeSE3ProjectDirect* edge = new EdgeSE3ProjectDirect ();
            edge->setVertex ( 0, pose );
            edge->camera_ = curr_->camera_.get();
            edge->point_ = m.pos_world;
            edge->image_ = &(curr_->gray_);
            edge->setMeasurement ( m.grayscale );
            edge->setInformation ( Eigen::Matrix<double,1,1>::Identity() );
            edge->setId ( index++ );
            optimizer.addEdge ( edge );
        }
        cout<<"edges in graph: "<<optimizer.edges().size() <<endl;
        optimizer.initializeOptimization();
        optimizer.optimize ( 30 );
        T_c_w_estimated_ = SE3 (
                pose->estimate().rotation(),
                pose->estimate().translation()
        );
        cout<<"T_c_w_estimated_: "<<endl<<T_c_w_estimated_.matrix()<<endl;
    }

}
