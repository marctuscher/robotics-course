//**************************************************************************
// Multi circle_tracker
//
// Track circles in an RGB image provided using Hought-Trransform
//
//**************************************************************************
// Date:    
// Author:   Ralf Gulde
// Version:  0.1
// Changes:  None
//**************************************************************************

#include <stdio.h>
#include "ros/ros.h"
#include <iostream>
#include "std_msgs/String.h"
#include "sensor_msgs/Image.h"
#include <cv_bridge/cv_bridge.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "std_msgs/Float64MultiArray.h"
#include "sensor_msgs/image_encodings.h"
#include "std_srvs/SetBool.h"

// initial and max values of the parameters of interests.
int cannyThresholdInitialValue_rgb = 0;
int accumulatorThresholdInitialValue_rgb = 0;
int minRadius_rgb = 0;
int maxRadius_rgb = 0;
double minDist_rgb = 0;

int cannyThresholdInitialValue_infra = 0;
int accumulatorThresholdInitialValue_infra = 0;
int minRadius_infra = 0;
int maxRadius_infra = 0;
double minDist_infra = 0;

using namespace cv;
using namespace std;

class SubscribeAndPublish{
     
  public:
    SubscribeAndPublish(){
    
    //Register Subscriber
    sub_rgb = n_.subscribe("/camera/color/image_raw", 1, &SubscribeAndPublish::callback_rgb, this);
    sub_infra_1 = n_.subscribe("/camera/infra1/image_rect_raw", 1, &SubscribeAndPublish::callback_infra_1, this);
    sub_infra_2 = n_.subscribe("/camera/infra2/image_rect_raw", 1, &SubscribeAndPublish::callback_infra_2, this);

    //Register Publisher
    pub_infra_1 = n_.advertise<std_msgs::Float64MultiArray>("/circle_tracker/infra1", 1);
    pub_infra_2 = n_.advertise<std_msgs::Float64MultiArray>("/circle_tracker/infra2", 1);
    pub_rgb = n_.advertise<std_msgs::Float64MultiArray>("/circle_tracker/rgb", 1);

  }
    
  void callback_rgb(const sensor_msgs::Image& msg){
	
    Mat img_mat, img_gray;
	  cv_bridge::CvImagePtr cv_ptr;
	
	  cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
	  img_mat = (cv_ptr->image).clone();
	 
	  // Convert it to gray
	  cvtColor( img_mat, img_gray, COLOR_BGR2GRAY );

  	// Reduce the noise so we avoid false circle detection
	  GaussianBlur( img_gray, img_gray, Size(9, 9), 2, 2 );
	
	  // will hold the results of theextern detection
    std::vector<Vec3f> circles;
	
    // runs the actual detection
    HoughCircles(img_gray, circles, CV_HOUGH_GRADIENT, 1, minDist_rgb, cannyThresholdInitialValue_rgb, accumulatorThresholdInitialValue_rgb, minRadius_rgb, maxRadius_rgb);	
	  //printf("Found circle: x: %f,y: %f\n", circles[0][0], circles[0][1]);
	
	  //--
    if(circles.size() > 0){
      std_msgs::Float64MultiArray circ_msg;

      for( size_t i = 0; i < circles.size(); i++ ){
        // x-coordinate
        circ_msg.data.push_back(circles[i][0]);
        // y-coordinate
        circ_msg.data.push_back(circles[i][1]);
        // radius
        circ_msg.data.push_back(circles[i][2]);
      }
      pub_rgb.publish(circ_msg);
      circ_msg.data.clear();
    }
  }

    
  void callback_infra_1(const sensor_msgs::Image& msg){
	
    Mat img_mat, img_gray;
	  cv_bridge::CvImagePtr cv_ptr;
	
	  cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
	  img_gray = (cv_ptr->image).clone();
	 
	  // Convert it to gray
	  //cvtColor( img_mat, img_gray, COLOR_BGR2GRAY );

  	// Reduce the noise so we avoid false circle detection
	  GaussianBlur( img_gray, img_gray, Size(9, 9), 2, 2 );
	
	  // will hold the results of theextern detection
    std::vector<Vec3f> circles;
	
    // runs the actual detection

    HoughCircles(img_gray, circles, CV_HOUGH_GRADIENT, 1, minDist_infra, cannyThresholdInitialValue_infra, accumulatorThresholdInitialValue_infra, minRadius_infra, maxRadius_infra);
	
	  //--
    if(circles.size() > 0){
      std_msgs::Float64MultiArray circ_msg;

      for( size_t i = 0; i < circles.size(); i++ ){
        // x-coordinate
        circ_msg.data.push_back(circles[i][0]);
        // y-coordinate
        circ_msg.data.push_back(circles[i][1]);
        // radius
        circ_msg.data.push_back(circles[i][2]);
      }
      pub_infra_1.publish(circ_msg);
      circ_msg.data.clear();
    }
  }
  
  void callback_infra_2(const sensor_msgs::Image& msg){
	
    Mat img_mat, img_gray;
	  cv_bridge::CvImagePtr cv_ptr;
	
	  cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
	  img_gray = (cv_ptr->image).clone();
	 
	  // Convert it to gray
	  //cvtColor( img_mat, img_gray, COLOR_BGR2GRAY );

  	// Reduce the noise so we avoid false circle detection
	  GaussianBlur( img_gray, img_gray, Size(9, 9), 2, 2 );
	
	  // will hold the results of theextern detection
    std::vector<Vec3f> circles;
	
    // runs the actual detection
    HoughCircles(img_gray, circles, CV_HOUGH_GRADIENT, 1, minDist_infra, cannyThresholdInitialValue_infra, accumulatorThresholdInitialValue_infra, minRadius_infra, maxRadius_infra);
	
	  //--
    if(circles.size() > 0){
      std_msgs::Float64MultiArray circ_msg;

      for( size_t i = 0; i < circles.size(); i++ ){
        // x-coordinate
        circ_msg.data.push_back(circles[i][0]);
        // y-coordinate
        circ_msg.data.push_back(circles[i][1]);
        // radius
        circ_msg.data.push_back(circles[i][2]);
      }
      pub_infra_2.publish(circ_msg);
      circ_msg.data.clear();
    }
  }
       
  /**
   * NodeHandle is the main access point to communications with the ROS system.
   * The first NodeHandle constructed will fully initialize this node, and the last
   * NodeHandle destructed will close down the node.
   */

  private:
    ros::NodeHandle n_; 
    ros::Publisher pub_rgb;
    ros::Publisher pub_infra_1;
    ros::Publisher pub_infra_2;
    ros::Subscriber sub_rgb;
    ros::Subscriber sub_infra_1;
    ros::Subscriber sub_infra_2;
    ros::ServiceServer proceedHOUGH;
    //ros::ServiceServer triggerSS;


}; //End of class SubscribeAndPublish


int main(int argc, char **argv){
   
  ros::init(argc, argv, "multi_circle_tracker");
  
  //*minDist:	minimum distance between the centers of the detected circles. 
  //  If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. 
  //  If it is too large, some circles may be missed.
  //*param1(canny): it is the higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller).
  //*param2(accu):	it is the accumulator threshold for the circle centers at the detection stage. 
  //  The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first.
  //*minRadius	minimum circle radius.
  //*maxRadius	maximum circle radius.

  cannyThresholdInitialValue_rgb = 60;
  accumulatorThresholdInitialValue_rgb = 18;
  minRadius_rgb = 40;
  maxRadius_rgb = 80;
  minDist_rgb = 30.0;

  cannyThresholdInitialValue_infra = 60;
  accumulatorThresholdInitialValue_infra = 18;
  minRadius_infra = 40;
  maxRadius_infra = 80;
  minDist_infra = 30.0;
  
  
  ROS_INFO("Performing multi circle tracking using Hough-tramsform...");
   
  //Create an object of class SubscribeAndPublish that will take care of everything
  SubscribeAndPublish SAPObject;
    
  /**
   * ros::spin() will enter a loop, pumping callbacks.  With this version, all
   * callbacks will be called from within this thread (the main one).  ros::spin()
   * will exit when Ctrl-C is pressed, or the node is shutdown by the master.
   */

  ros::spin();
  return 0;
}

