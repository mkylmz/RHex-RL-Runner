#include <stdlib.h>
#include <cstdint>
#include <set>
#include <raisim/OgreVis.hpp>
#include "RaisimGymEnv.hpp"
#include "visSetupCallback.hpp"
#include <math.h>

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
    rhex_ = world_->addArticulatedSystem(resourceDir+"/urdf/RHex.urdf");
    rhex_->setControlMode(raisim::ControlMode::FORCE_AND_TORQUE);
    auto ground = world_->addGround();
    world_->setTimeStep(simulation_dt_);
    world_->setERP(simulation_dt_,simulation_dt_);
    world_->setDefaultMaterial(10.0, 0.0, 0.0);
    /// get robot data
    gcDim_ = rhex_->getGeneralizedCoordinateDim(); // will be six; joint angles
    gvDim_ = rhex_->getDOF(); // will be six; angular velocity of joints.
    nJoints_ = 6;
    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_); gc_init_[2]=0.25; gc_init_[3]=1.0; gc_init_[7] = 3.14; gc_init_[11] = 3.14; gc_init_[15] = 3.14;
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    rhex_->setGeneralizedForce(Eigen::VectorXd::Zero(nJoints_));

    /// set pd gains
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_);
    jointPgain.setZero(nJoints_); jointPgain.setConstant(40.0);
    jointDgain.setZero(nJoints_); jointDgain.setConstant(2.1);
    target_torques.setZero(gvDim_);
    rhex_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

     /* Convention
      *
      *   observation space = [ height                       n =  1, cur_tot =  0
      *                         body Linear velocities,      n =  3, cur_tot =  1
      *                         body Angular velocities,     n =  3, cur_tot =  4
      *                         joint angles,                n =  6, cur_tot =  7
      *                         joint velocities,            n =  6, cur_tot = 13 ] total 19
      */

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 20; /// convention described above
    actionDim_ = nJoints_;
    actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obMean_.setZero(obDim_); obStd_.setZero(obDim_);
    time_passed = 0.;

    /// action & observation scaling
    actionMean_ = Eigen::VectorXd::Constant(6, -0.8);
    actionStd_.setConstant(0.7);

    obMean_ << 0.26, /// average height 1
        Eigen::VectorXd::Constant(1, 1.0), /// body lin x vel 1
        Eigen::VectorXd::Constant(5, 0.0), /// body lin/ang vel 5
        Eigen::VectorXd::Constant(1, 0.0), /// joint position 1
        Eigen::VectorXd::Constant(1, 3.14),/// joint position 1
        Eigen::VectorXd::Constant(1, 0.0), /// joint position 1
        Eigen::VectorXd::Constant(1, 3.14),/// joint position 1
        Eigen::VectorXd::Constant(1, 0.0), /// joint position 1
        Eigen::VectorXd::Constant(1, 3.14),/// joint position 1
        Eigen::VectorXd::Constant(6, 20.0),/// joint vel history 6
        Eigen::VectorXd::Constant(1, 0.0); /// time phase 1

    obStd_ << 1.0, /// average height
        Eigen::VectorXd::Constant(3, 1.0), /// linear velocity
        Eigen::VectorXd::Constant(3, 1.0), /// angular velocities
        Eigen::VectorXd::Constant(6, 1.0), /// joint angles
        Eigen::VectorXd::Constant(6, 1.0), /// joint velocities
        Eigen::VectorXd::Constant(1, 1.0); /// time phase

    /// Reward coefficients
    READ_YAML(double, forwardVelRewardCoeff_, cfg["forwardVelRewardCoeff"])
    READ_YAML(double, torqueRewardCoeff_, cfg["torqueRewardCoeff"])

    gui::rewardLogger.init({"forwardVelReward", "torqueReward"});

    footIndices_.insert(rhex_->getBodyIdx("link_leg1Upper"));
    footIndices_.insert(rhex_->getBodyIdx("link_leg2Upper"));
    footIndices_.insert(rhex_->getBodyIdx("link_leg3Upper"));
    footIndices_.insert(rhex_->getBodyIdx("link_leg4Upper"));
    footIndices_.insert(rhex_->getBodyIdx("link_leg5Upper"));
    footIndices_.insert(rhex_->getBodyIdx("link_leg6Upper"));
    footIndices_.insert(rhex_->getBodyIdx("link_leg1Lower"));
    footIndices_.insert(rhex_->getBodyIdx("link_leg2Lower"));
    footIndices_.insert(rhex_->getBodyIdx("link_leg3Lower"));
    footIndices_.insert(rhex_->getBodyIdx("link_leg4Lower"));
    footIndices_.insert(rhex_->getBodyIdx("link_leg5Lower"));
    footIndices_.insert(rhex_->getBodyIdx("link_leg6Lower"));

    /// visualize if it is the first environment
    if (visualizable_) {
      cout<<"Visualization initializing..."<<endl;
      auto vis = raisim::OgreVis::get();

      /// these method must be called before initApp
      vis->setWorld(world_.get());
      vis->setWindowSize(1280, 720);
      vis->setImguiSetupCallback(imguiSetupCallback);
      vis->setImguiRenderCallback(imguiRenderCallBack);
      vis->setKeyboardCallback(raisimKeyboardCallback);
      vis->setSetUpCallback(setupCallback);
      vis->setAntiAliasing(2);

      /// starts visualizer thread
      vis->initApp();

      rhexVisual_ = vis->createGraphicalObject(rhex_, "RHex");
      vis->createGraphicalObject(ground, 20, "floor", "checkerboard_green");
      desired_fps_ = 60.;
      vis->setDesiredFPS(desired_fps_);
      cout<<"Visualization initialized."<<endl;
    }
    cout<<"Created Environment."<<endl;
  }

  ~ENVIRONMENT() final = default;


  void init() final { }

  void reset() final {
    rhex_->setState(gc_init_, gv_init_);
    auto force = rhex_->getGeneralizedForce();
    force.setZero();
    rhex_->setGeneralizedForce(force);
    world_start_time = world_->getWorldTime();
    time_passed = 0.;
    
    updateObservation();
    if(visualizable_)
      gui::rewardLogger.clean();
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    pTarget6_ = action.cast<double>();
    pTarget6_ = pTarget6_.cwiseProduct(actionStd_);
    pTarget6_ += actionMean_;

    //rhex_->setPdTarget(pTarget_, vTarget_);
    auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);
    auto visDecimation = int(1. / (desired_fps_ * simulation_dt_) + 1e-10);

    for(int i=0; i<loopCount; i++) {
      
      gc_ = rhex_->getGeneralizedCoordinate();
      gv_ = rhex_->getGeneralizedVelocity();
      for (uint j=0; j<6; j++)
      {
        float poserr   = pTarget6_[j];
        float speederr = -gv_[6+2*j];
        
        // The allowable error range is determined by the error offset
        // parameter: range = [-PI+offset,PI+offset].
        if ( poserr <= - M_PI - M_PI/2+M_PI/36 ) poserr += 2*M_PI;
        if ( poserr > M_PI - M_PI/2+M_PI/36 ) poserr -= 2*M_PI;

        // Compute torque command
        float hipCommTorque = ( jointPgain[j] * poserr + jointDgain[j] * speederr );
        if (hipCommTorque > 20.0)
        {
          target_torques[6+2*j] = 20.0;
          hipCommTorque = 20.0;
        }
        else if (hipCommTorque < -20.0)
        {
          target_torques[6+2*j] = -20.0;
          hipCommTorque = -20.0;
        }
        else
          target_torques[6+2*j] = hipCommTorque;
      }
      rhex_->setGeneralizedForce(target_torques);

      world_->integrate();

      if (visualizable_ && visualizeThisStep_ && visualizationCounter_ % visDecimation == 0)
        raisim::OgreVis::get()->renderOneFrame();

      visualizationCounter_++;
    }

    updateObservation();

    torqueReward_ = torqueRewardCoeff_ * rhex_->getGeneralizedForce().squaredNorm();
    forwardVelReward_ = forwardVelRewardCoeff_ * bodyLinearVel_[0];

    if(visualizeThisStep_) {
      gui::rewardLogger.log("torqueReward", torqueReward_);
      gui::rewardLogger.log("forwardVelReward", forwardVelReward_);

      /// reset camera
      auto vis = raisim::OgreVis::get();

      vis->select(rhexVisual_->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.3), 3, true);
    }

    return torqueReward_ + forwardVelReward_;
  }

  void updateExtraInfo() final {
    extraInfo_["forward vel reward"] = forwardVelReward_;
    extraInfo_["base height"] = gc_[2];
  }

  void updateObservation() {
    //rhex_->getState(gc_, gv_);
    gc_ = rhex_->getGeneralizedCoordinate();
    gv_ = rhex_->getGeneralizedVelocity();
    obDouble_.setZero(obDim_); obScaled_.setZero(obDim_);

    /// body height
    obDouble_[0] = gc_[2];

    /// body orientation
    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    //obDouble_.segment(1, 3) = rot.e().row(2);

    /// body velocities
    bodyLinearVel_[0] = gv_[0];  bodyLinearVel_[1] = gv_[1];  bodyLinearVel_[2] = gv_[2];
    bodyAngularVel_[0] = gv_[3];  bodyAngularVel_[1] = gv_[4];  bodyAngularVel_[2] = gv_[5];
    obDouble_.segment(1, 3) = bodyLinearVel_;
    obDouble_.segment(4, 3) = bodyAngularVel_;

    /// joint angles and velocities
    for (uint i=0; i<6; i++)
    {
      obDouble_[7+i] = gc_[7+2*i];
      obDouble_[13+i] = gv_[6+2*i];
    }

    // Time phase
    time_passed = world_->getWorldTime() - world_start_time;
    obDouble_[19] = fmod(time_passed,0.3);
    //printf("Phase: %f, world time: %f, start_time: %f\n", obDouble_[19],  world_->getWorldTime(), world_start_time);

    obScaled_ = (obDouble_-obMean_).cwiseQuotient(obStd_);
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obScaled_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_);

    /// if the contact body is not feet
    for(auto& contact: rhex_->getContacts())
      if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end()) {
        return true;
      }

    terminalReward = 0.f;
    return false;
  }

  void setSeed(int seed) final {
    std::srand(seed);
  }

  void close() final {
  }

  private:
    int gcDim_, gvDim_, nJoints_;
    double reward_;
    bool visualizable_ = false;
    std::normal_distribution<double> distribution_;
    raisim::ArticulatedSystem* rhex_;
    std::vector<GraphicObject> * rhexVisual_;
    Eigen::VectorXd gc_init_, gv_init_, pTarget_, pTarget6_, vTarget_, torque_;
    raisim::VecDyn gc_, gv_;
    Eigen::VectorXd jointPgain, jointDgain, target_torques;
    double terminalRewardCoeff_ = -10.;
    double forwardVelRewardCoeff_ = 0., forwardVelReward_ = 0.;
    double torqueRewardCoeff_ = 0., torqueReward_ = 0.;
    double desired_fps_ = 60.;
    double time_passed = 0., world_start_time = 0.;
    int visualizationCounter_=0;
    Eigen::VectorXd actionMean_, actionStd_, obMean_, obStd_;
    Eigen::VectorXd obDouble_, obScaled_;
    Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
    std::set<size_t> footIndices_;
  };

}