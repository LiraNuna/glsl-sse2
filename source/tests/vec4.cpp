#include "vec4.h"
#include "../vec4.h"

namespace tests
{

void vec4::testEquality()
{
	::vec4 v(1, 2, 3, 4);

	assert(v == v);
	assert(v == ::vec4(1, 2, 3, 4));
	assert(::vec4(1, 2, 3, 4) == v);
	assert(::vec4(1, 2, 3, 4) == ::vec4(1, 2, 3, 4));
}

void vec4::testSwizzleEquality()
{
	::vec4 v(1, 2, 3, 4);

	assert(v.wzyx == ::vec4(4, 3, 2, 1));
	assert(::vec4(4, 3, 2, 1) == v.wzyx);

	assert(v.wzyx.wzyx == v);
	assert(v.xyzw.xyzw == v);
	assert(v.xxxx.yyyy == ::vec4(1));
	assert(v.yyyy.xxxx == ::vec4(2));
}

}
