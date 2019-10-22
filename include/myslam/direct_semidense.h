#ifndef MYSLAM_DIRECT_SEMIDENSE_H
#define MYSLAM_DIRECT_SEMIDENSE_H
#include "myslam/common_include.h"

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

#ifndef VISUALODOMETRY_H
#define VISUALODOMETRY_H

#include "myslam/common_include.h"
#include "myslam/frame.h"

#include <opencv2/features2d/features2d.hpp>

namespace myslam
{
class DirectMethod
{
public: // functions
    DirectMethod();
    ~DirectMethod();

    bool directInit( Frame::Ptr frame );



};

}



#endif //MYSLAM_DIRECT_SEMIDENSE_H
