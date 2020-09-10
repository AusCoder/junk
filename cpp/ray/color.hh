#ifndef _COLOR_HH
#define _COLOR_HH

#include "vec3.hh"
#include <ostream>

void write_color(std::ostream &out, color c) {
    out << static_cast<int>(255.999 * c.x()) << ' '
        << static_cast<int>(255.999 * c.y()) << ' '
        << static_cast<int>(255.999 * c.z()) << '\n';
}

#endif