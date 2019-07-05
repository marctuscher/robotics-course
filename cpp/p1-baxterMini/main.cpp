#include <Kin/kin.h>
#include <RosCom/baxter.h>
#include <Operate/robotOperation.h>

void minimal_use(){
  //load a configuration
  rai::KinematicWorld C;
  C.addFile("../../rai-robotModels/baxter/baxter.g");

  //define a home and zero pose
  arr q_home = C.getJointState();
  arr q_zero = 0.*q_home;

  //launch the interface
  BaxterInterface B(true);

  for(uint i=0;i<40;i++){
    rai::wait(.1);
    B.send_q(q_home); //repeatedly send q_home as reference -> moves
    cout <<B.get_q() <<endl;
    cout <<B.get_qdot() <<endl;
    cout <<B.get_u() <<endl;
  }

  //just once send q_zero as reference -> will hardly move
  B.send_q(q_zero);

  C.setJointState(q_zero);
  C.watch(true);
}


void spline_use(){
  //load a configuration
  rai::KinematicWorld C;
  C.addFile("../../rai-robotModels/baxter/baxter.g");

  //define a home and zero pose
  arr q_home = C.getJointState();
  arr q_zero = 0.*q_home;
  arr q_home_cycle = {-0.04410195, -0.34936412, 0.65692727, -0.41417481, -0.19328158, 1.11098559, -1.19535453, 1.06496616, 1.31807299, -0.53650978, 0.35856801, 1.14550015, 0.96180595, 1.14665064, -0.3117816, 0.0 ,0.0};

  //launch the interface
  RobotOperation B(C);
  cout <<"joint names: " <<B.getJointNames() <<endl;
  B.sendToReal(true);

  
  for (int i = 0; i < 40; i++){
  B.move({q_home}, {4.});
  rai::wait(4.);
  B.move({q_home_cycle}, {4.});
  rai::wait(4.);

  }

  //output states
  for(;;){
    cout <<" q:" <<B.getJointPositions()
        <<" gripper right:" <<B.getGripperOpened("right") <<' ' <<B.getGripperGrabbed("right")
       <<" gripper left:" <<B.getGripperOpened("left") <<' ' <<B.getGripperGrabbed("left")
      <<endl;
    if(!B.timeToGo()) break;
    rai::wait(.1);
  }
  cout <<"motion done!" <<endl;
  rai::wait();

  //instantaneous move of the reference (baxter does interpolation)
  B.moveHard(q_home);
  rai::wait();

  //close right gripper
  q_home(-2) = 1.;
  B.moveHard(q_home);
  rai::wait();

  //open right gripper
  q_home(-2) = 0.;
  B.moveHard(q_home);
  rai::wait();

  //close left gripper
  q_home(-1) = 1.;
  B.moveHard(q_home);
  rai::wait();

  //open left gripper
  q_home(-1) = 0.;
  B.moveHard(q_home);
  rai::wait();
}


int main(int argc,char **argv){
  rai::initCmdLine(argc,argv);

//  minimal_use();

putenv("ROS_MASTER_URI=http://thecount.local:11311"); 
putenv("ROS_IP=129.69.216.200");
  spline_use();

  return 0;
}
