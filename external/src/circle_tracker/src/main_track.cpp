//**************************************************************************
// Circle_tracker
//
// Track circles in an RGB image provided by the Kinect v2 using Hought-Trransform
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
const bool fullHD = 1;

int cannyThresholdInitialValue = 0;
int accumulatorThresholdInitialValue = 0;
int minRadius = 0;
int maxRadius = 0;
double minDist = 0;

using namespace cv;
using namespace std;

class SubscribeAndPublish{
     
  public:
    SubscribeAndPublish(){
    
    fun_setMode(0);
    //Topic to publish 
    pub_ = n_.advertise<std_msgs::Float64MultiArray>("/circle_tracker/circles", 1);


    //triggerSS = n_.advertiseService("kinect_trigger/trigger", &SubscribeAndPublish::triggerSrv, this);
    proceedHOUGH = n_.advertiseService("circle_tracker/setMode", &SubscribeAndPublish::setModeSrv, this);

  }
    
  void fun_setMode(int m){
    mode = m;
  }
  
    //bool setModeSrv(cob_srvs::SetInt::Request& req, cob_srvs::SetInt::Response& res){
  bool setModeSrv(std_srvs::SetBool::Request& req, std_srvs::SetBool::Response& res){
	  switch(req.data) {

		  case 0: 
		    ROS_INFO("Disabling circle-tracker!");
		    fun_setMode(0);
		    
		    //Un-Subscribe topic
		    sub_.shutdown();
		    
		    res.success = true;
		    res.message = "Disabling circle-tracker!";
		    return true;
		  break;
		
		  case 1: 
		    ROS_INFO("Activating circle-tracker!");
		    fun_setMode(1);
		    
		    
		    //Subscribe topic
		    if(fullHD){
			    sub_ = n_.subscribe("/camera/color/image_raw", 1, &SubscribeAndPublish::callback, this);
		    }else{
			    sub_ = n_.subscribe("/camera/color/image_raw", 1, &SubscribeAndPublish::callback, this);
		    }
		    
		    
		    res.success = true;
		    res.message = "Activating circle-tracker!";
		    return true;
		  break;
		  
		  default: 
		    ROS_INFO("Mode '%d' unknown. Mode '%d'unchanged", req.data, mode);
		    res.success = false;
		    res.message = "Mode '%d' unknown. Mode '%d'unchanged", req.data, mode;
		    return false;
		  break;
	  }
	}
    
    
  void callback(const sensor_msgs::Image& msg){
	
    Mat img_mat, img_gray;
	  cv_bridge::CvImagePtr cv_ptr;
	
	  cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
	  img_mat = (cv_ptr->image).clone();
	
	 
	  // Convert it to gray
	  cvtColor( img_mat, img_gray, COLOR_BGR2GRAY );

  	// Reduce the noise so we avoid false circle detection
	  GaussianBlur( img_gray, img_gray, Size(9, 9), 2, 2 );
	
	  // will hold the results of the detection
    std::vector<Vec3f> circles;
	
    // runs the actual detection

    HoughCircles( img_gray, circles, CV_HOUGH_GRADIENT, 1, minDist, cannyThresholdInitialValue, accumulatorThresholdInitialValue, minRadius, maxRadius);
	
	  //printf("Found circle: x: %f,y: %f\n", circles[0][0], circles[0][1]);
	
	  //--
	
	  std_msgs::Float64MultiArray circ_msg;

	  for( size_t i = 0; i < circles.size(); i++ ){
	    // x-coordinate
      circ_msg.data.push_back(circles[i][0]);
	    // y-coordinate
      circ_msg.data.push_back(circles[i][1]);
	    // radius
      circ_msg.data.push_back(circles[i][2]);
    }
    pub_.publish(circ_msg);
    circ_msg.data.clear();
  }
    
    
  /**
   * NodeHandle is the main access point to communications with the ROS system.
   * The first NodeHandle constructed will fully initialize this node, and the last
   * NodeHandle destructed will close down the node.
   */

  private:
    ros::NodeHandle n_; 
    ros::Publisher pub_;
    ros::Subscriber sub_;
    ros::ServiceServer proceedHOUGH;
    int mode;
    //ros::ServiceServer triggerSS;


}; //End of class SubscribeAndPublish



int main(int argc, char **argv){
   
    
  /**
   * The ros::init() function needs to see argc and argv so that it can perform
   * any ROS arguments and name remapping that were provided at the command line.
   * For programmatic remappings you can use a different version of init() which takes
   * remappings directly, but for most command-line programs, passing argc and argv is
   * the easiest way to do it.  The third argument to init() is the name of the node.
   *
   * You must call one of the versions of ros::init() before using any other
   * part of the ROS system.
   */
  ros::init(argc, argv, "circle_tracker");
  

  
  if(fullHD){
    
    // set up Prameters for fullHD RGB Stream
    accumulatorThresholdInitialValue = 20;
    cannyThresholdInitialValue = 100;
    minRadius = 0;
    maxRadius = 0;
    minDist = 30.0;
    
  }else{
    // set up Prameters for quarter-HD RGB Stream
    accumulatorThresholdInitialValue = 40;
    cannyThresholdInitialValue = 160;
    minRadius = 7;
    maxRadius = 50;
    minDist = 5.0;
    
  }
  
  
  ROS_INFO("Performing circle tracking using Hough-tramsform...");
   
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

