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
		static void testSwizzleRead();
		static void testSwizzleWrite();
		static void testSwizzleReadWrite();

		static void testAccessors();
};

}

#endif /* TESTS_VEC4_H */
