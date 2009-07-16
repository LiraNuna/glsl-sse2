#include "vec4.h"
#include "../vec4.h"

#include <cmath>

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

	assert(v.xyzw == v);
	assert(v.yyyy == v.yyyy);
	assert(v.wzzw == v.wzzw);

	assert(v.xyzw.xyzw == v);
	assert(v.wzyx.wzyx == v);

	assert(v.yzwx.yzwx == v.zwxy);
	assert(v.xwzy.xwzy == v.xyzw);

	assert(v.xxxx.yyyy == ::vec4(1));
	assert(v.yyyy.xxxx == ::vec4(2));
	assert(v.wwww.xyzw == ::vec4(4));

	assert(v.zzzz.xyzw == ::vec4(3));
	assert(v.wwww.yyzz == ::vec4(4));
}

void vec4::testSwizzleAccessors()
{
	::vec4 v(1, 2, 3, 4);
/*
	assert(v.xyzw.x == v.x);
	assert(v.xyzw.y == v.y);

	assert(v.wzyx.x == v.w);
	assert(v.wzyx.y == v.z);
*/
}

void vec4::testSwizzleWrite()
{
	::vec4 v;

	v.xyzw = ::vec4(1, 2, 3, 4);
	assert(v == ::vec4(1, 2, 3, 4));

	v.wzyx = ::vec4(1, 2, 3, 4);
	assert(v == ::vec4(4, 3, 2, 1));

	v.yzwx = ::vec4(1, 2, 3, 4);
	assert(v == ::vec4(4, 1, 2, 3));

	v.yxzw = ::vec4(1, 2, 3, 4);
	assert(v == ::vec4(2, 1, 3, 4));

	v.zywx = ::vec4(1,2,3,4);
	assert(v == ::vec4(4, 2, 1, 3));

	v.zywx = ::vec4(1,2,3,4).xyzw;
	assert(v == ::vec4(4, 2, 1, 3));

	v.zywx = ::vec4(1,2,3,4).wxyz;
	assert(v == ::vec4(3, 1, 4, 2));

	v.yzwx = ::vec4(1, 2, 3, 4).yzwx;
	assert(v == ::vec4(1, 2, 3, 4));
}

void vec4::testUnary()
{
	::vec4 v(-1337, 42, -0, 85070591730234615865843651857942052864.F);	// bignum = 2**126

	::vec4 v_sqrt = sqrt(v);
	assert(std::isnan(v_sqrt.x));
	assert(approxEqual(v_sqrt.y, 6.4807407F));
	assert(v_sqrt.z == 0);
	assert(v_sqrt.w == 9223372036854775808.F);	// bignum = 2**63
}

void vec4::testAccessors()
{
	::vec4 v(1, 2, 3, 4);

	assert(v.x == 1);
	assert(v.y == 2);
	assert(v.z == 3);
	assert(v.w == 4);

	assert(v.r == 1);
	assert(v.g == 2);
	assert(v.b == 3);
	assert(v.a == 4);

	assert(v.s == 1);
	assert(v.t == 2);
	assert(v.p == 3);
	assert(v.q == 4);

	assert(v[0] == 1);
	assert(v[1] == 2);
	assert(v[2] == 3);
	assert(v[3] == 4);

	assert(&v.y == &v.x + 1);
	assert(&v.z == &v.y + 1);
	assert(&v.w == &v.z + 1);

	assert(&v.g == &v.r + 1);
	assert(&v.b == &v.g + 1);
	assert(&v.a == &v.b + 1);

	assert(&v.t == &v.s + 1);
	assert(&v.p == &v.t + 1);
	assert(&v.q == &v.p + 1);

	assert(&v.x == &v.r);
	assert(&v.y == &v.g);
	assert(&v.z == &v.b);
	assert(&v.w == &v.a);

	assert(&v.x == &v.s);
	assert(&v.y == &v.t);
	assert(&v.z == &v.p);
	assert(&v.w == &v.q);

	float *f = v;

	assert(f[0] == 1);
	assert(f[1] == 2);
	assert(f[2] == 3);
	assert(f[3] == 4);

	assert(sizeof(v) == 4 * sizeof(float));
}

template<typename T>
bool vec4::approxEqual(T a, T b, T fuzziness)
{
	T diff = a - b;
	return std::abs(diff) <= fuzziness;
}

}
