#ifndef TESTS_VEC4_H
#define TESTS_VEC4_H

#include "test.h"

namespace tests
{

class vec4 :
	public test
{
	public:
		static void testEquality();
		static void testInequality();

		static void testSwizzleEquality();
		static void testSwizzleAccessors();
		static void testSwizzleRead();
		static void testSwizzleWrite();
		static void testSwizzleReadWrite();

		static void testUnary();

		static void testAccessors();

	protected:
		template<typename T>
		static bool approxEqual(T a, T b, T fuzziness = 1.0F / (1 << 20));
};

}

#endif /* TESTS_VEC4_H */
