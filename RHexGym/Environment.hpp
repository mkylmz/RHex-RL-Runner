#include <stdlib.h>
#include <cstdint>
#include <set>
#include <raisim/OgreVis.hpp>
#include "RaisimGymEnv.hpp"
#include "visSetupCallback.hpp"

#include "visualizer/raisimKeyboardCallback.hpp"
#include "visualizer/helper.hpp"
#include "visualizer/guiState.hpp"
#include "visualizer/raisimBasicImguiPanel.hpp"

using namespace std;
#define deg2rad(ang) ((ang) * M_PI / 180.0)
#define rad2deg(ang) ((ang) * 180.0 / M_PI)


namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

  public:

  explicit ENVIRONMENT(const std::string& resourceDir, const YAML::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), distribution_(0.0, 0.2), visualizable_(visualizable) {

    /// add objects
    cout<<resourceDir<<endl;
    rhex_ = world_->addArticulatedSystem(resourceDir+"/RHex.urdf");
    rhex_->setControlMode(raisim::ControlMode::FORCE_AND_TORQUE);
    auto ground = world_->addGround();
    world_->setERP(0,0);
    /// get robot data
    gcDim_ = rhex_->getGeneralizedCoordinateDim(); // will be six; joint angles
    gvDim_ = rhex_->getDOF(); // will be six; angular velocity of joints.
    nJoints_ = 6;
    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    rhex_->setGeneralizedForce(Eigen::VectorXd::Zero(nJoints_));

     /* Convention
      *
      *   observation space = [ body Linear velocities,      n =  3, cur_tot =  0
      *                         body Angular velocities,     n =  3, cur_tot =  3
      *                         joint angles,                n =  6, cur_tot =  6
      *                         joint velocities,            n =  6, cur_tot = 12 ] total 18
      */

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 18; /// convention described above
    actionDim_ = nJoints_;
    actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obMean_.setZero(obDim_); obStd_.setZero(obDim_);

    /// action & observation scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(0.6);

    /** TO DO : CONTINUE **/

  }

  private:
    int gcDim_, gvDim_, nJoints_;
    double reward_;
    bool visualizable_ = false;
    std::normal_distribution<double> distribution_;
    raisim::ArticulatedSystem* rhex_;
    std::vector<GraphicObject> * rhexVisual_;
    Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, actionScaled_,torque_;
    double terminalRewardCoeff_ = -10.;
    double forceRewardCoeff_ = 0., forceReward_ = 0.;
    double desired_fps_ = 60.;
    int visualizationCounter_=0;
    Eigen::VectorXd actionMean_, actionStd_, obMean_, obStd_;
    Eigen::VectorXd obDouble_, obScaled_;
  };

}