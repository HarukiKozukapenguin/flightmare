#pragma once

#include <stdlib.h>
#include "flightlib/common/types.hpp"

namespace flightlib {

class Teleport {
 public:
    Scalar t;
    Vector<3> tree_pos;
    bool is_teleport;
    bool direction;
    void reset(){
        t = 0.0;
        tree_pos.setZero();
        is_teleport = false;
        direction = false;
    }
};
} // namespace flightlibb