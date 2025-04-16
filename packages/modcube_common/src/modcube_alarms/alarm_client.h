#include <mutex>
#include <set>
#include <vector>
#include <ros/ros.h>
#include <modcube_msgs/AlarmReport.h>
#include "alarms.h"

#pragma once

namespace modcube_alarms {
  class AlarmClient {
      public:
          AlarmClient(ros::NodeHandle& n);
          void set(modcube_alarms::AlarmType type, const std::string &msg, bool value = true);
          void clear(modcube_alarms::AlarmType type, const std::string &msg);
          bool check(modcube_alarms::AlarmType type);
      private:
          ros::NodeHandle &n;

          std::set<modcube_alarms::AlarmType> active_alarms;

          ros::Time last_update_time;
          ros::Duration timeout;

          std::mutex lock;

          ros::Subscriber report_sub;
          ros::ServiceClient sync_srv;

          void handle_report(const modcube_msgs::AlarmReport::ConstPtr &msg);

          void set_active_alarms(std::vector<int> alarms);
  };
}
