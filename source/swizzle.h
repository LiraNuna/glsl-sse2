#ifndef __SWIZZLE_H__
#define __SWIZZLE_H__

template<typename S, typename V>
struct _swizzle4_maker
{
		// Merges mask `target` with `m` into one unified mask that does the same sequential shuffle
	template <unsigned target, unsigned m>
	struct _mask_merger
	{
		enum
		{
			ROW0 = ((target >> (((m >> 0) & 3) << 1)) & 3) << 0,
			ROW1 = ((target >> (((m >> 2) & 3) << 1)) & 3) << 2,
			ROW2 = ((target >> (((m >> 4) & 3) << 1)) & 3) << 4,
			ROW3 = ((target >> (((m >> 6) & 3) << 1)) & 3) << 6,

			MASK = ROW0 | ROW1 | ROW2 | ROW3,
		};

		private:
			_mask_merger();
	};

		// Since we are working in little endian land, this reverses the shuffle mask
	template <unsigned m>
	struct _mask_reverser
	{
		enum
		{
			ROW0 = 0 << (((m >> 0) & 3) << 1),
			ROW1 = 1 << (((m >> 2) & 3) << 1),
			ROW2 = 2 << (((m >> 4) & 3) << 1),
			ROW3 = 3 << (((m >> 6) & 3) << 1),

			MASK = ROW0 | ROW1 | ROW2 | ROW3,
		};

		private:
			_mask_reverser();
	};

	template<typename TV, typename TS, unsigned mask>
	struct base
	{
			// This massive contructor maps a vector to references
		inline base(TV &v):
			x(v[(mask >> 0) & 0x3]), y(v[(mask >> 2) & 0x3]),
			z(v[(mask >> 4) & 0x3]), w(v[(mask >> 6) & 0x3]),

			r(v[(mask >> 0) & 0x3]), g(v[(mask >> 2) & 0x3]),
			b(v[(mask >> 4) & 0x3]), a(v[(mask >> 6) & 0x3]),

			s(v[(mask >> 0) & 0x3]), t(v[(mask >> 2) & 0x3]),
			p(v[(mask >> 4) & 0x3]), q(v[(mask >> 6) & 0x3]),

			v(v) {
				// Empty
		}

		// ----------------------------------------------------------------- //

		inline operator const V () const {
			return __shuffled(v.m);
		}

		inline TS operator[](int index) {
			return v[(mask >> (index << 1)) & 0x3];
		}

		// ----------------------------------------------------------------- //

		TS x, y, z, w;
		TS r, g, b, a;
		TS s, t, p, q;

			// Ideally this should be protected
			// but since we dwell into cross-template arcane magic...
		TV v;

		protected:
			inline const __m128 __shuffle(const __m128 &m, const int s_mask) const {
				return _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m), s_mask));
			}

			inline const __m128 __shuffle(const __m128i &m, const int s_mask) const {
				return _mm_shuffle_epi32(m, s_mask);
			}

		private:
			inline const V __shuffled(const __m128 &m) const {
				return _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(m), mask));
			}

			inline const V __shuffled(const __m128i &m) const {
				return _mm_shuffle_epi32(m, mask);
			}
	};

		// Read only
	template<unsigned mask>
	struct ro : public base<const V &, const S &, mask>
	{
		inline ro(const V &v) : base<const V &, const S &, mask>(v) {
			// Empty
		}

			// Swizzle of the swizzle, read only const
		template<unsigned other_mask>
		inline ro<_mask_merger<mask, other_mask>::MASK> shuffle4_ro() const {
			typedef _mask_merger<mask, other_mask> merged;

			return ro<merged::MASK>(this->v);
		}

			// Swizzle of the swizzle, read/write const
		template<unsigned other_mask>
		inline ro<_mask_merger<mask, other_mask>::MASK> shuffle4_rw() const {
			typedef _mask_merger<mask, other_mask> merged;

			return ro<merged::MASK>(this->v);
		}
	};

		// Read / Write
	template<unsigned mask>
	struct rw : public base<V &, S &, mask>
	{
		inline rw(V &v) : base<V &, S &, mask>(v) {
			// Empty
		}

		// ----------------------------------------------------------------- //

			// Swizzle from V
		inline rw& operator = (const V &v) {
			this->v.m = __shuffle(v.m, _mask_reverser<mask>::MASK);
			return *this;
		}

			// Swizzle from same r/o mask (v1.xyzw = v2.xyzw)
		inline V& operator = (const ro<mask> &s) {
			this->v.m = s.v.m;
			return this->v;
		}

			// Swizzle from same mask (v1.xyzw = v2.xyzw)
		inline V& operator = (const rw<mask> &s) {
			this->v.m = s.v.m;
			return this->v;
		}

		// ----------------------------------------------------------------- //

			// Swizzle mask => other_mask, r/o (v1.zwxy = v2.xyxy)
		template<unsigned other_mask>
		inline rw& operator = (const ro<other_mask> &s) {
			typedef _mask_merger<other_mask, _mask_reverser<mask>::MASK> merged;
			this->v.m = __shuffle(s.v.m, merged::MASK);
			return *this;
		}

			// Swizzle mask => other_mask (v1.zwxy = v2.zwxy)
		template<unsigned other_mask>
		inline rw<_mask_merger<other_mask, _mask_reverser<mask>::MASK>::MASK> operator = (const rw<other_mask> &s) {
			typedef _mask_merger<other_mask, _mask_reverser<mask>::MASK> merged;
			this->v.m = __shuffle(s.v.m, merged::MASK);
			return rw<merged::MASK>(this->v);
		}

		// ----------------------------------------------------------------- //

		inline V& operator += (S s) {
			return this->v += s;
		}

		inline V& operator += (const V &v) {
			return this->v = this->v.shuffle4_ro<mask>() + v;
		}

		inline V& operator -= (S s) {
			return this->v -= s;
		}

		inline V& operator -= (const V &v) {
			return this->v = this->v.shuffle4_ro<mask>() - v;
		}

		inline V& operator *= (S s) {
			return this->v *= s;
		}

		inline V& operator *= (const V &v) {
			return this->v = this->v.shuffle4_ro<mask>() * v;
		}

		inline V& operator /= (S s) {
			return this->v /= s;
		}

		inline V& operator /= (const V &v) {
			return this->v = this->v.shuffle4_ro<mask>() / v;
		}

		// ----------------------------------------------------------------- //

			// Swizzle of the swizzle, read only (v.zyxw.yyyy)
		template<unsigned other_mask>
		inline ro<_mask_merger<mask, other_mask>::MASK> shuffle4_ro() const {
			typedef _mask_merger<mask, other_mask> merged;

			return ro<merged::MASK>(this->v);
		}

			// Swizzle of the swizzle, read/write (v1.zyxw.wzyx)
		template<unsigned other_mask>
		inline rw<_mask_merger<mask, other_mask>::MASK> shuffle4_rw() {
			typedef _mask_merger<mask, other_mask> merged;

			return rw<merged::MASK>(this->v);
		}
	};
};

#define wzyx shuffle4_rw<_MM_SHUFFLE(0,1,2,3)>()
#define zwyx shuffle4_rw<_MM_SHUFFLE(0,1,3,2)>()
#define wyzx shuffle4_rw<_MM_SHUFFLE(0,2,1,3)>()
#define ywzx shuffle4_rw<_MM_SHUFFLE(0,2,3,1)>()
#define zywx shuffle4_rw<_MM_SHUFFLE(0,3,1,2)>()
#define yzwx shuffle4_rw<_MM_SHUFFLE(0,3,2,1)>()
#define wzxy shuffle4_rw<_MM_SHUFFLE(1,0,2,3)>()
#define zwxy shuffle4_rw<_MM_SHUFFLE(1,0,3,2)>()
#define wxzy shuffle4_rw<_MM_SHUFFLE(1,2,0,3)>()
#define xwzy shuffle4_rw<_MM_SHUFFLE(1,2,3,0)>()
#define zxwy shuffle4_rw<_MM_SHUFFLE(1,3,0,2)>()
#define xzwy shuffle4_rw<_MM_SHUFFLE(1,3,2,0)>()
#define wyxz shuffle4_rw<_MM_SHUFFLE(2,0,1,3)>()
#define ywxz shuffle4_rw<_MM_SHUFFLE(2,0,3,1)>()
#define wxyz shuffle4_rw<_MM_SHUFFLE(2,1,0,3)>()
#define xwyz shuffle4_rw<_MM_SHUFFLE(2,1,3,0)>()
#define yxwz shuffle4_rw<_MM_SHUFFLE(2,3,0,1)>()
#define xywz shuffle4_rw<_MM_SHUFFLE(2,3,1,0)>()
#define zyxw shuffle4_rw<_MM_SHUFFLE(3,0,1,2)>()
#define yzxw shuffle4_rw<_MM_SHUFFLE(3,0,2,1)>()
#define zxyw shuffle4_rw<_MM_SHUFFLE(3,1,0,2)>()
#define xzyw shuffle4_rw<_MM_SHUFFLE(3,1,2,0)>()
#define yxzw shuffle4_rw<_MM_SHUFFLE(3,2,0,1)>()
#define xyzw shuffle4_rw<_MM_SHUFFLE(3,2,1,0)>()

#define xxxx shuffle4_ro<_MM_SHUFFLE(0,0,0,0)>()
#define yxxx shuffle4_ro<_MM_SHUFFLE(0,0,0,1)>()
#define zxxx shuffle4_ro<_MM_SHUFFLE(0,0,0,2)>()
#define wxxx shuffle4_ro<_MM_SHUFFLE(0,0,0,3)>()
#define xyxx shuffle4_ro<_MM_SHUFFLE(0,0,1,0)>()
#define yyxx shuffle4_ro<_MM_SHUFFLE(0,0,1,1)>()
#define zyxx shuffle4_ro<_MM_SHUFFLE(0,0,1,2)>()
#define wyxx shuffle4_ro<_MM_SHUFFLE(0,0,1,3)>()
#define xzxx shuffle4_ro<_MM_SHUFFLE(0,0,2,0)>()
#define yzxx shuffle4_ro<_MM_SHUFFLE(0,0,2,1)>()
#define zzxx shuffle4_ro<_MM_SHUFFLE(0,0,2,2)>()
#define wzxx shuffle4_ro<_MM_SHUFFLE(0,0,2,3)>()
#define xwxx shuffle4_ro<_MM_SHUFFLE(0,0,3,0)>()
#define ywxx shuffle4_ro<_MM_SHUFFLE(0,0,3,1)>()
#define zwxx shuffle4_ro<_MM_SHUFFLE(0,0,3,2)>()
#define wwxx shuffle4_ro<_MM_SHUFFLE(0,0,3,3)>()
#define xxyx shuffle4_ro<_MM_SHUFFLE(0,1,0,0)>()
#define yxyx shuffle4_ro<_MM_SHUFFLE(0,1,0,1)>()
#define zxyx shuffle4_ro<_MM_SHUFFLE(0,1,0,2)>()
#define wxyx shuffle4_ro<_MM_SHUFFLE(0,1,0,3)>()
#define xyyx shuffle4_ro<_MM_SHUFFLE(0,1,1,0)>()
#define yyyx shuffle4_ro<_MM_SHUFFLE(0,1,1,1)>()
#define zyyx shuffle4_ro<_MM_SHUFFLE(0,1,1,2)>()
#define wyyx shuffle4_ro<_MM_SHUFFLE(0,1,1,3)>()
#define xzyx shuffle4_ro<_MM_SHUFFLE(0,1,2,0)>()
#define yzyx shuffle4_ro<_MM_SHUFFLE(0,1,2,1)>()
#define zzyx shuffle4_ro<_MM_SHUFFLE(0,1,2,2)>()
#define xwyx shuffle4_ro<_MM_SHUFFLE(0,1,3,0)>()
#define ywyx shuffle4_ro<_MM_SHUFFLE(0,1,3,1)>()
#define wwyx shuffle4_ro<_MM_SHUFFLE(0,1,3,3)>()
#define xxzx shuffle4_ro<_MM_SHUFFLE(0,2,0,0)>()
#define yxzx shuffle4_ro<_MM_SHUFFLE(0,2,0,1)>()
#define zxzx shuffle4_ro<_MM_SHUFFLE(0,2,0,2)>()
#define wxzx shuffle4_ro<_MM_SHUFFLE(0,2,0,3)>()
#define xyzx shuffle4_ro<_MM_SHUFFLE(0,2,1,0)>()
#define yyzx shuffle4_ro<_MM_SHUFFLE(0,2,1,1)>()
#define zyzx shuffle4_ro<_MM_SHUFFLE(0,2,1,2)>()
#define xzzx shuffle4_ro<_MM_SHUFFLE(0,2,2,0)>()
#define yzzx shuffle4_ro<_MM_SHUFFLE(0,2,2,1)>()
#define zzzx shuffle4_ro<_MM_SHUFFLE(0,2,2,2)>()
#define wzzx shuffle4_ro<_MM_SHUFFLE(0,2,2,3)>()
#define xwzx shuffle4_ro<_MM_SHUFFLE(0,2,3,0)>()
#define zwzx shuffle4_ro<_MM_SHUFFLE(0,2,3,2)>()
#define wwzx shuffle4_ro<_MM_SHUFFLE(0,2,3,3)>()
#define xxwx shuffle4_ro<_MM_SHUFFLE(0,3,0,0)>()
#define yxwx shuffle4_ro<_MM_SHUFFLE(0,3,0,1)>()
#define zxwx shuffle4_ro<_MM_SHUFFLE(0,3,0,2)>()
#define wxwx shuffle4_ro<_MM_SHUFFLE(0,3,0,3)>()
#define xywx shuffle4_ro<_MM_SHUFFLE(0,3,1,0)>()
#define yywx shuffle4_ro<_MM_SHUFFLE(0,3,1,1)>()
#define wywx shuffle4_ro<_MM_SHUFFLE(0,3,1,3)>()
#define xzwx shuffle4_ro<_MM_SHUFFLE(0,3,2,0)>()
#define zzwx shuffle4_ro<_MM_SHUFFLE(0,3,2,2)>()
#define wzwx shuffle4_ro<_MM_SHUFFLE(0,3,2,3)>()
#define xwwx shuffle4_ro<_MM_SHUFFLE(0,3,3,0)>()
#define ywwx shuffle4_ro<_MM_SHUFFLE(0,3,3,1)>()
#define zwwx shuffle4_ro<_MM_SHUFFLE(0,3,3,2)>()
#define wwwx shuffle4_ro<_MM_SHUFFLE(0,3,3,3)>()
#define xxxy shuffle4_ro<_MM_SHUFFLE(1,0,0,0)>()
#define yxxy shuffle4_ro<_MM_SHUFFLE(1,0,0,1)>()
#define zxxy shuffle4_ro<_MM_SHUFFLE(1,0,0,2)>()
#define wxxy shuffle4_ro<_MM_SHUFFLE(1,0,0,3)>()
#define xyxy shuffle4_ro<_MM_SHUFFLE(1,0,1,0)>()
#define yyxy shuffle4_ro<_MM_SHUFFLE(1,0,1,1)>()
#define zyxy shuffle4_ro<_MM_SHUFFLE(1,0,1,2)>()
#define wyxy shuffle4_ro<_MM_SHUFFLE(1,0,1,3)>()
#define xzxy shuffle4_ro<_MM_SHUFFLE(1,0,2,0)>()
#define yzxy shuffle4_ro<_MM_SHUFFLE(1,0,2,1)>()
#define zzxy shuffle4_ro<_MM_SHUFFLE(1,0,2,2)>()
#define xwxy shuffle4_ro<_MM_SHUFFLE(1,0,3,0)>()
#define ywxy shuffle4_ro<_MM_SHUFFLE(1,0,3,1)>()
#define wwxy shuffle4_ro<_MM_SHUFFLE(1,0,3,3)>()
#define xxyy shuffle4_ro<_MM_SHUFFLE(1,1,0,0)>()
#define yxyy shuffle4_ro<_MM_SHUFFLE(1,1,0,1)>()
#define zxyy shuffle4_ro<_MM_SHUFFLE(1,1,0,2)>()
#define wxyy shuffle4_ro<_MM_SHUFFLE(1,1,0,3)>()
#define xyyy shuffle4_ro<_MM_SHUFFLE(1,1,1,0)>()
#define yyyy shuffle4_ro<_MM_SHUFFLE(1,1,1,1)>()
#define zyyy shuffle4_ro<_MM_SHUFFLE(1,1,1,2)>()
#define wyyy shuffle4_ro<_MM_SHUFFLE(1,1,1,3)>()
#define xzyy shuffle4_ro<_MM_SHUFFLE(1,1,2,0)>()
#define yzyy shuffle4_ro<_MM_SHUFFLE(1,1,2,1)>()
#define zzyy shuffle4_ro<_MM_SHUFFLE(1,1,2,2)>()
#define wzyy shuffle4_ro<_MM_SHUFFLE(1,1,2,3)>()
#define xwyy shuffle4_ro<_MM_SHUFFLE(1,1,3,0)>()
#define ywyy shuffle4_ro<_MM_SHUFFLE(1,1,3,1)>()
#define zwyy shuffle4_ro<_MM_SHUFFLE(1,1,3,2)>()
#define wwyy shuffle4_ro<_MM_SHUFFLE(1,1,3,3)>()
#define xxzy shuffle4_ro<_MM_SHUFFLE(1,2,0,0)>()
#define yxzy shuffle4_ro<_MM_SHUFFLE(1,2,0,1)>()
#define zxzy shuffle4_ro<_MM_SHUFFLE(1,2,0,2)>()
#define xyzy shuffle4_ro<_MM_SHUFFLE(1,2,1,0)>()
#define yyzy shuffle4_ro<_MM_SHUFFLE(1,2,1,1)>()
#define zyzy shuffle4_ro<_MM_SHUFFLE(1,2,1,2)>()
#define wyzy shuffle4_ro<_MM_SHUFFLE(1,2,1,3)>()
#define xzzy shuffle4_ro<_MM_SHUFFLE(1,2,2,0)>()
#define yzzy shuffle4_ro<_MM_SHUFFLE(1,2,2,1)>()
#define zzzy shuffle4_ro<_MM_SHUFFLE(1,2,2,2)>()
#define wzzy shuffle4_ro<_MM_SHUFFLE(1,2,2,3)>()
#define ywzy shuffle4_ro<_MM_SHUFFLE(1,2,3,1)>()
#define zwzy shuffle4_ro<_MM_SHUFFLE(1,2,3,2)>()
#define wwzy shuffle4_ro<_MM_SHUFFLE(1,2,3,3)>()
#define xxwy shuffle4_ro<_MM_SHUFFLE(1,3,0,0)>()
#define yxwy shuffle4_ro<_MM_SHUFFLE(1,3,0,1)>()
#define wxwy shuffle4_ro<_MM_SHUFFLE(1,3,0,3)>()
#define xywy shuffle4_ro<_MM_SHUFFLE(1,3,1,0)>()
#define yywy shuffle4_ro<_MM_SHUFFLE(1,3,1,1)>()
#define zywy shuffle4_ro<_MM_SHUFFLE(1,3,1,2)>()
#define wywy shuffle4_ro<_MM_SHUFFLE(1,3,1,3)>()
#define yzwy shuffle4_ro<_MM_SHUFFLE(1,3,2,1)>()
#define zzwy shuffle4_ro<_MM_SHUFFLE(1,3,2,2)>()
#define wzwy shuffle4_ro<_MM_SHUFFLE(1,3,2,3)>()
#define xwwy shuffle4_ro<_MM_SHUFFLE(1,3,3,0)>()
#define ywwy shuffle4_ro<_MM_SHUFFLE(1,3,3,1)>()
#define zwwy shuffle4_ro<_MM_SHUFFLE(1,3,3,2)>()
#define wwwy shuffle4_ro<_MM_SHUFFLE(1,3,3,3)>()
#define xxxz shuffle4_ro<_MM_SHUFFLE(2,0,0,0)>()
#define yxxz shuffle4_ro<_MM_SHUFFLE(2,0,0,1)>()
#define zxxz shuffle4_ro<_MM_SHUFFLE(2,0,0,2)>()
#define wxxz shuffle4_ro<_MM_SHUFFLE(2,0,0,3)>()
#define xyxz shuffle4_ro<_MM_SHUFFLE(2,0,1,0)>()
#define yyxz shuffle4_ro<_MM_SHUFFLE(2,0,1,1)>()
#define zyxz shuffle4_ro<_MM_SHUFFLE(2,0,1,2)>()
#define xzxz shuffle4_ro<_MM_SHUFFLE(2,0,2,0)>()
#define yzxz shuffle4_ro<_MM_SHUFFLE(2,0,2,1)>()
#define zzxz shuffle4_ro<_MM_SHUFFLE(2,0,2,2)>()
#define wzxz shuffle4_ro<_MM_SHUFFLE(2,0,2,3)>()
#define xwxz shuffle4_ro<_MM_SHUFFLE(2,0,3,0)>()
#define zwxz shuffle4_ro<_MM_SHUFFLE(2,0,3,2)>()
#define wwxz shuffle4_ro<_MM_SHUFFLE(2,0,3,3)>()
#define xxyz shuffle4_ro<_MM_SHUFFLE(2,1,0,0)>()
#define yxyz shuffle4_ro<_MM_SHUFFLE(2,1,0,1)>()
#define zxyz shuffle4_ro<_MM_SHUFFLE(2,1,0,2)>()
#define xyyz shuffle4_ro<_MM_SHUFFLE(2,1,1,0)>()
#define yyyz shuffle4_ro<_MM_SHUFFLE(2,1,1,1)>()
#define zyyz shuffle4_ro<_MM_SHUFFLE(2,1,1,2)>()
#define wyyz shuffle4_ro<_MM_SHUFFLE(2,1,1,3)>()
#define xzyz shuffle4_ro<_MM_SHUFFLE(2,1,2,0)>()
#define yzyz shuffle4_ro<_MM_SHUFFLE(2,1,2,1)>()
#define zzyz shuffle4_ro<_MM_SHUFFLE(2,1,2,2)>()
#define wzyz shuffle4_ro<_MM_SHUFFLE(2,1,2,3)>()
#define ywyz shuffle4_ro<_MM_SHUFFLE(2,1,3,1)>()
#define zwyz shuffle4_ro<_MM_SHUFFLE(2,1,3,2)>()
#define wwyz shuffle4_ro<_MM_SHUFFLE(2,1,3,3)>()
#define xxzz shuffle4_ro<_MM_SHUFFLE(2,2,0,0)>()
#define yxzz shuffle4_ro<_MM_SHUFFLE(2,2,0,1)>()
#define zxzz shuffle4_ro<_MM_SHUFFLE(2,2,0,2)>()
#define wxzz shuffle4_ro<_MM_SHUFFLE(2,2,0,3)>()
#define xyzz shuffle4_ro<_MM_SHUFFLE(2,2,1,0)>()
#define yyzz shuffle4_ro<_MM_SHUFFLE(2,2,1,1)>()
#define zyzz shuffle4_ro<_MM_SHUFFLE(2,2,1,2)>()
#define wyzz shuffle4_ro<_MM_SHUFFLE(2,2,1,3)>()
#define xzzz shuffle4_ro<_MM_SHUFFLE(2,2,2,0)>()
#define yzzz shuffle4_ro<_MM_SHUFFLE(2,2,2,1)>()
#define zzzz shuffle4_ro<_MM_SHUFFLE(2,2,2,2)>()
#define wzzz shuffle4_ro<_MM_SHUFFLE(2,2,2,3)>()
#define xwzz shuffle4_ro<_MM_SHUFFLE(2,2,3,0)>()
#define ywzz shuffle4_ro<_MM_SHUFFLE(2,2,3,1)>()
#define zwzz shuffle4_ro<_MM_SHUFFLE(2,2,3,2)>()
#define wwzz shuffle4_ro<_MM_SHUFFLE(2,2,3,3)>()
#define xxwz shuffle4_ro<_MM_SHUFFLE(2,3,0,0)>()
#define zxwz shuffle4_ro<_MM_SHUFFLE(2,3,0,2)>()
#define wxwz shuffle4_ro<_MM_SHUFFLE(2,3,0,3)>()
#define yywz shuffle4_ro<_MM_SHUFFLE(2,3,1,1)>()
#define zywz shuffle4_ro<_MM_SHUFFLE(2,3,1,2)>()
#define wywz shuffle4_ro<_MM_SHUFFLE(2,3,1,3)>()
#define xzwz shuffle4_ro<_MM_SHUFFLE(2,3,2,0)>()
#define yzwz shuffle4_ro<_MM_SHUFFLE(2,3,2,1)>()
#define zzwz shuffle4_ro<_MM_SHUFFLE(2,3,2,2)>()
#define wzwz shuffle4_ro<_MM_SHUFFLE(2,3,2,3)>()
#define xwwz shuffle4_ro<_MM_SHUFFLE(2,3,3,0)>()
#define ywwz shuffle4_ro<_MM_SHUFFLE(2,3,3,1)>()
#define zwwz shuffle4_ro<_MM_SHUFFLE(2,3,3,2)>()
#define wwwz shuffle4_ro<_MM_SHUFFLE(2,3,3,3)>()
#define xxxw shuffle4_ro<_MM_SHUFFLE(3,0,0,0)>()
#define yxxw shuffle4_ro<_MM_SHUFFLE(3,0,0,1)>()
#define zxxw shuffle4_ro<_MM_SHUFFLE(3,0,0,2)>()
#define wxxw shuffle4_ro<_MM_SHUFFLE(3,0,0,3)>()
#define xyxw shuffle4_ro<_MM_SHUFFLE(3,0,1,0)>()
#define yyxw shuffle4_ro<_MM_SHUFFLE(3,0,1,1)>()
#define wyxw shuffle4_ro<_MM_SHUFFLE(3,0,1,3)>()
#define xzxw shuffle4_ro<_MM_SHUFFLE(3,0,2,0)>()
#define zzxw shuffle4_ro<_MM_SHUFFLE(3,0,2,2)>()
#define wzxw shuffle4_ro<_MM_SHUFFLE(3,0,2,3)>()
#define xwxw shuffle4_ro<_MM_SHUFFLE(3,0,3,0)>()
#define ywxw shuffle4_ro<_MM_SHUFFLE(3,0,3,1)>()
#define zwxw shuffle4_ro<_MM_SHUFFLE(3,0,3,2)>()
#define wwxw shuffle4_ro<_MM_SHUFFLE(3,0,3,3)>()
#define xxyw shuffle4_ro<_MM_SHUFFLE(3,1,0,0)>()
#define yxyw shuffle4_ro<_MM_SHUFFLE(3,1,0,1)>()
#define wxyw shuffle4_ro<_MM_SHUFFLE(3,1,0,3)>()
#define xyyw shuffle4_ro<_MM_SHUFFLE(3,1,1,0)>()
#define yyyw shuffle4_ro<_MM_SHUFFLE(3,1,1,1)>()
#define zyyw shuffle4_ro<_MM_SHUFFLE(3,1,1,2)>()
#define wyyw shuffle4_ro<_MM_SHUFFLE(3,1,1,3)>()
#define yzyw shuffle4_ro<_MM_SHUFFLE(3,1,2,1)>()
#define zzyw shuffle4_ro<_MM_SHUFFLE(3,1,2,2)>()
#define wzyw shuffle4_ro<_MM_SHUFFLE(3,1,2,3)>()
#define xwyw shuffle4_ro<_MM_SHUFFLE(3,1,3,0)>()
#define ywyw shuffle4_ro<_MM_SHUFFLE(3,1,3,1)>()
#define zwyw shuffle4_ro<_MM_SHUFFLE(3,1,3,2)>()
#define wwyw shuffle4_ro<_MM_SHUFFLE(3,1,3,3)>()
#define xxzw shuffle4_ro<_MM_SHUFFLE(3,2,0,0)>()
#define zxzw shuffle4_ro<_MM_SHUFFLE(3,2,0,2)>()
#define wxzw shuffle4_ro<_MM_SHUFFLE(3,2,0,3)>()
#define yyzw shuffle4_ro<_MM_SHUFFLE(3,2,1,1)>()
#define zyzw shuffle4_ro<_MM_SHUFFLE(3,2,1,2)>()
#define wyzw shuffle4_ro<_MM_SHUFFLE(3,2,1,3)>()
#define xzzw shuffle4_ro<_MM_SHUFFLE(3,2,2,0)>()
#define yzzw shuffle4_ro<_MM_SHUFFLE(3,2,2,1)>()
#define zzzw shuffle4_ro<_MM_SHUFFLE(3,2,2,2)>()
#define wzzw shuffle4_ro<_MM_SHUFFLE(3,2,2,3)>()
#define xwzw shuffle4_ro<_MM_SHUFFLE(3,2,3,0)>()
#define ywzw shuffle4_ro<_MM_SHUFFLE(3,2,3,1)>()
#define zwzw shuffle4_ro<_MM_SHUFFLE(3,2,3,2)>()
#define wwzw shuffle4_ro<_MM_SHUFFLE(3,2,3,3)>()
#define xxww shuffle4_ro<_MM_SHUFFLE(3,3,0,0)>()
#define yxww shuffle4_ro<_MM_SHUFFLE(3,3,0,1)>()
#define zxww shuffle4_ro<_MM_SHUFFLE(3,3,0,2)>()
#define wxww shuffle4_ro<_MM_SHUFFLE(3,3,0,3)>()
#define xyww shuffle4_ro<_MM_SHUFFLE(3,3,1,0)>()
#define yyww shuffle4_ro<_MM_SHUFFLE(3,3,1,1)>()
#define zyww shuffle4_ro<_MM_SHUFFLE(3,3,1,2)>()
#define wyww shuffle4_ro<_MM_SHUFFLE(3,3,1,3)>()
#define xzww shuffle4_ro<_MM_SHUFFLE(3,3,2,0)>()
#define yzww shuffle4_ro<_MM_SHUFFLE(3,3,2,1)>()
#define zzww shuffle4_ro<_MM_SHUFFLE(3,3,2,2)>()
#define wzww shuffle4_ro<_MM_SHUFFLE(3,3,2,3)>()
#define xwww shuffle4_ro<_MM_SHUFFLE(3,3,3,0)>()
#define ywww shuffle4_ro<_MM_SHUFFLE(3,3,3,1)>()
#define zwww shuffle4_ro<_MM_SHUFFLE(3,3,3,2)>()
#define wwww shuffle4_ro<_MM_SHUFFLE(3,3,3,3)>()

// -------------------------------------------------------------------------- //

#define abgr shuffle4_rw<_MM_SHUFFLE(0,1,2,3)>()
#define bagr shuffle4_rw<_MM_SHUFFLE(0,1,3,2)>()
#define agbr shuffle4_rw<_MM_SHUFFLE(0,2,1,3)>()
#define gabr shuffle4_rw<_MM_SHUFFLE(0,2,3,1)>()
#define bgar shuffle4_rw<_MM_SHUFFLE(0,3,1,2)>()
#define gbar shuffle4_rw<_MM_SHUFFLE(0,3,2,1)>()
#define abrg shuffle4_rw<_MM_SHUFFLE(1,0,2,3)>()
#define barg shuffle4_rw<_MM_SHUFFLE(1,0,3,2)>()
#define arbg shuffle4_rw<_MM_SHUFFLE(1,2,0,3)>()
#define rabg shuffle4_rw<_MM_SHUFFLE(1,2,3,0)>()
#define brag shuffle4_rw<_MM_SHUFFLE(1,3,0,2)>()
#define rbag shuffle4_rw<_MM_SHUFFLE(1,3,2,0)>()
#define agrb shuffle4_rw<_MM_SHUFFLE(2,0,1,3)>()
#define garb shuffle4_rw<_MM_SHUFFLE(2,0,3,1)>()
#define argb shuffle4_rw<_MM_SHUFFLE(2,1,0,3)>()
#define ragb shuffle4_rw<_MM_SHUFFLE(2,1,3,0)>()
#define grab shuffle4_rw<_MM_SHUFFLE(2,3,0,1)>()
#define rgab shuffle4_rw<_MM_SHUFFLE(2,3,1,0)>()
#define bgra shuffle4_rw<_MM_SHUFFLE(3,0,1,2)>()
#define gbra shuffle4_rw<_MM_SHUFFLE(3,0,2,1)>()
#define brga shuffle4_rw<_MM_SHUFFLE(3,1,0,2)>()
#define rbga shuffle4_rw<_MM_SHUFFLE(3,1,2,0)>()
#define grba shuffle4_rw<_MM_SHUFFLE(3,2,0,1)>()
#define rgba shuffle4_rw<_MM_SHUFFLE(3,2,1,0)>()

#define rrrr shuffle4_ro<_MM_SHUFFLE(0,0,0,0)>()
#define grrr shuffle4_ro<_MM_SHUFFLE(0,0,0,1)>()
#define brrr shuffle4_ro<_MM_SHUFFLE(0,0,0,2)>()
#define arrr shuffle4_ro<_MM_SHUFFLE(0,0,0,3)>()
#define rgrr shuffle4_ro<_MM_SHUFFLE(0,0,1,0)>()
#define ggrr shuffle4_ro<_MM_SHUFFLE(0,0,1,1)>()
#define bgrr shuffle4_ro<_MM_SHUFFLE(0,0,1,2)>()
#define agrr shuffle4_ro<_MM_SHUFFLE(0,0,1,3)>()
#define rbrr shuffle4_ro<_MM_SHUFFLE(0,0,2,0)>()
#define gbrr shuffle4_ro<_MM_SHUFFLE(0,0,2,1)>()
#define bbrr shuffle4_ro<_MM_SHUFFLE(0,0,2,2)>()
#define abrr shuffle4_ro<_MM_SHUFFLE(0,0,2,3)>()
#define rarr shuffle4_ro<_MM_SHUFFLE(0,0,3,0)>()
#define garr shuffle4_ro<_MM_SHUFFLE(0,0,3,1)>()
#define barr shuffle4_ro<_MM_SHUFFLE(0,0,3,2)>()
#define aarr shuffle4_ro<_MM_SHUFFLE(0,0,3,3)>()
#define rrgr shuffle4_ro<_MM_SHUFFLE(0,1,0,0)>()
#define grgr shuffle4_ro<_MM_SHUFFLE(0,1,0,1)>()
#define brgr shuffle4_ro<_MM_SHUFFLE(0,1,0,2)>()
#define argr shuffle4_ro<_MM_SHUFFLE(0,1,0,3)>()
#define rggr shuffle4_ro<_MM_SHUFFLE(0,1,1,0)>()
#define gggr shuffle4_ro<_MM_SHUFFLE(0,1,1,1)>()
#define bggr shuffle4_ro<_MM_SHUFFLE(0,1,1,2)>()
#define aggr shuffle4_ro<_MM_SHUFFLE(0,1,1,3)>()
#define rbgr shuffle4_ro<_MM_SHUFFLE(0,1,2,0)>()
#define gbgr shuffle4_ro<_MM_SHUFFLE(0,1,2,1)>()
#define bbgr shuffle4_ro<_MM_SHUFFLE(0,1,2,2)>()
#define ragr shuffle4_ro<_MM_SHUFFLE(0,1,3,0)>()
#define gagr shuffle4_ro<_MM_SHUFFLE(0,1,3,1)>()
#define aagr shuffle4_ro<_MM_SHUFFLE(0,1,3,3)>()
#define rrbr shuffle4_ro<_MM_SHUFFLE(0,2,0,0)>()
#define grbr shuffle4_ro<_MM_SHUFFLE(0,2,0,1)>()
#define brbr shuffle4_ro<_MM_SHUFFLE(0,2,0,2)>()
#define arbr shuffle4_ro<_MM_SHUFFLE(0,2,0,3)>()
#define rgbr shuffle4_ro<_MM_SHUFFLE(0,2,1,0)>()
#define ggbr shuffle4_ro<_MM_SHUFFLE(0,2,1,1)>()
#define bgbr shuffle4_ro<_MM_SHUFFLE(0,2,1,2)>()
#define rbbr shuffle4_ro<_MM_SHUFFLE(0,2,2,0)>()
#define gbbr shuffle4_ro<_MM_SHUFFLE(0,2,2,1)>()
#define bbbr shuffle4_ro<_MM_SHUFFLE(0,2,2,2)>()
#define abbr shuffle4_ro<_MM_SHUFFLE(0,2,2,3)>()
#define rabr shuffle4_ro<_MM_SHUFFLE(0,2,3,0)>()
#define babr shuffle4_ro<_MM_SHUFFLE(0,2,3,2)>()
#define aabr shuffle4_ro<_MM_SHUFFLE(0,2,3,3)>()
#define rrar shuffle4_ro<_MM_SHUFFLE(0,3,0,0)>()
#define grar shuffle4_ro<_MM_SHUFFLE(0,3,0,1)>()
#define brar shuffle4_ro<_MM_SHUFFLE(0,3,0,2)>()
#define arar shuffle4_ro<_MM_SHUFFLE(0,3,0,3)>()
#define rgar shuffle4_ro<_MM_SHUFFLE(0,3,1,0)>()
#define ggar shuffle4_ro<_MM_SHUFFLE(0,3,1,1)>()
#define agar shuffle4_ro<_MM_SHUFFLE(0,3,1,3)>()
#define rbar shuffle4_ro<_MM_SHUFFLE(0,3,2,0)>()
#define bbar shuffle4_ro<_MM_SHUFFLE(0,3,2,2)>()
#define abar shuffle4_ro<_MM_SHUFFLE(0,3,2,3)>()
#define raar shuffle4_ro<_MM_SHUFFLE(0,3,3,0)>()
#define gaar shuffle4_ro<_MM_SHUFFLE(0,3,3,1)>()
#define baar shuffle4_ro<_MM_SHUFFLE(0,3,3,2)>()
#define aaar shuffle4_ro<_MM_SHUFFLE(0,3,3,3)>()
#define rrrg shuffle4_ro<_MM_SHUFFLE(1,0,0,0)>()
#define grrg shuffle4_ro<_MM_SHUFFLE(1,0,0,1)>()
#define brrg shuffle4_ro<_MM_SHUFFLE(1,0,0,2)>()
#define arrg shuffle4_ro<_MM_SHUFFLE(1,0,0,3)>()
#define rgrg shuffle4_ro<_MM_SHUFFLE(1,0,1,0)>()
#define ggrg shuffle4_ro<_MM_SHUFFLE(1,0,1,1)>()
#define bgrg shuffle4_ro<_MM_SHUFFLE(1,0,1,2)>()
#define agrg shuffle4_ro<_MM_SHUFFLE(1,0,1,3)>()
#define rbrg shuffle4_ro<_MM_SHUFFLE(1,0,2,0)>()
#define gbrg shuffle4_ro<_MM_SHUFFLE(1,0,2,1)>()
#define bbrg shuffle4_ro<_MM_SHUFFLE(1,0,2,2)>()
#define rarg shuffle4_ro<_MM_SHUFFLE(1,0,3,0)>()
#define garg shuffle4_ro<_MM_SHUFFLE(1,0,3,1)>()
#define aarg shuffle4_ro<_MM_SHUFFLE(1,0,3,3)>()
#define rrgg shuffle4_ro<_MM_SHUFFLE(1,1,0,0)>()
#define grgg shuffle4_ro<_MM_SHUFFLE(1,1,0,1)>()
#define brgg shuffle4_ro<_MM_SHUFFLE(1,1,0,2)>()
#define argg shuffle4_ro<_MM_SHUFFLE(1,1,0,3)>()
#define rggg shuffle4_ro<_MM_SHUFFLE(1,1,1,0)>()
#define gggg shuffle4_ro<_MM_SHUFFLE(1,1,1,1)>()
#define bggg shuffle4_ro<_MM_SHUFFLE(1,1,1,2)>()
#define aggg shuffle4_ro<_MM_SHUFFLE(1,1,1,3)>()
#define rbgg shuffle4_ro<_MM_SHUFFLE(1,1,2,0)>()
#define gbgg shuffle4_ro<_MM_SHUFFLE(1,1,2,1)>()
#define bbgg shuffle4_ro<_MM_SHUFFLE(1,1,2,2)>()
#define abgg shuffle4_ro<_MM_SHUFFLE(1,1,2,3)>()
#define ragg shuffle4_ro<_MM_SHUFFLE(1,1,3,0)>()
#define gagg shuffle4_ro<_MM_SHUFFLE(1,1,3,1)>()
#define bagg shuffle4_ro<_MM_SHUFFLE(1,1,3,2)>()
#define aagg shuffle4_ro<_MM_SHUFFLE(1,1,3,3)>()
#define rrbg shuffle4_ro<_MM_SHUFFLE(1,2,0,0)>()
#define grbg shuffle4_ro<_MM_SHUFFLE(1,2,0,1)>()
#define brbg shuffle4_ro<_MM_SHUFFLE(1,2,0,2)>()
#define rgbg shuffle4_ro<_MM_SHUFFLE(1,2,1,0)>()
#define ggbg shuffle4_ro<_MM_SHUFFLE(1,2,1,1)>()
#define bgbg shuffle4_ro<_MM_SHUFFLE(1,2,1,2)>()
#define agbg shuffle4_ro<_MM_SHUFFLE(1,2,1,3)>()
#define rbbg shuffle4_ro<_MM_SHUFFLE(1,2,2,0)>()
#define gbbg shuffle4_ro<_MM_SHUFFLE(1,2,2,1)>()
#define bbbg shuffle4_ro<_MM_SHUFFLE(1,2,2,2)>()
#define abbg shuffle4_ro<_MM_SHUFFLE(1,2,2,3)>()
#define gabg shuffle4_ro<_MM_SHUFFLE(1,2,3,1)>()
#define babg shuffle4_ro<_MM_SHUFFLE(1,2,3,2)>()
#define aabg shuffle4_ro<_MM_SHUFFLE(1,2,3,3)>()
#define rrag shuffle4_ro<_MM_SHUFFLE(1,3,0,0)>()
#define grag shuffle4_ro<_MM_SHUFFLE(1,3,0,1)>()
#define arag shuffle4_ro<_MM_SHUFFLE(1,3,0,3)>()
#define rgag shuffle4_ro<_MM_SHUFFLE(1,3,1,0)>()
#define ggag shuffle4_ro<_MM_SHUFFLE(1,3,1,1)>()
#define bgag shuffle4_ro<_MM_SHUFFLE(1,3,1,2)>()
#define agag shuffle4_ro<_MM_SHUFFLE(1,3,1,3)>()
#define gbag shuffle4_ro<_MM_SHUFFLE(1,3,2,1)>()
#define bbag shuffle4_ro<_MM_SHUFFLE(1,3,2,2)>()
#define abag shuffle4_ro<_MM_SHUFFLE(1,3,2,3)>()
#define raag shuffle4_ro<_MM_SHUFFLE(1,3,3,0)>()
#define gaag shuffle4_ro<_MM_SHUFFLE(1,3,3,1)>()
#define baag shuffle4_ro<_MM_SHUFFLE(1,3,3,2)>()
#define aaag shuffle4_ro<_MM_SHUFFLE(1,3,3,3)>()
#define rrrb shuffle4_ro<_MM_SHUFFLE(2,0,0,0)>()
#define grrb shuffle4_ro<_MM_SHUFFLE(2,0,0,1)>()
#define brrb shuffle4_ro<_MM_SHUFFLE(2,0,0,2)>()
#define arrb shuffle4_ro<_MM_SHUFFLE(2,0,0,3)>()
#define rgrb shuffle4_ro<_MM_SHUFFLE(2,0,1,0)>()
#define ggrb shuffle4_ro<_MM_SHUFFLE(2,0,1,1)>()
#define bgrb shuffle4_ro<_MM_SHUFFLE(2,0,1,2)>()
#define rbrb shuffle4_ro<_MM_SHUFFLE(2,0,2,0)>()
#define gbrb shuffle4_ro<_MM_SHUFFLE(2,0,2,1)>()
#define bbrb shuffle4_ro<_MM_SHUFFLE(2,0,2,2)>()
#define abrb shuffle4_ro<_MM_SHUFFLE(2,0,2,3)>()
#define rarb shuffle4_ro<_MM_SHUFFLE(2,0,3,0)>()
#define barb shuffle4_ro<_MM_SHUFFLE(2,0,3,2)>()
#define aarb shuffle4_ro<_MM_SHUFFLE(2,0,3,3)>()
#define rrgb shuffle4_ro<_MM_SHUFFLE(2,1,0,0)>()
#define grgb shuffle4_ro<_MM_SHUFFLE(2,1,0,1)>()
#define brgb shuffle4_ro<_MM_SHUFFLE(2,1,0,2)>()
#define rggb shuffle4_ro<_MM_SHUFFLE(2,1,1,0)>()
#define gggb shuffle4_ro<_MM_SHUFFLE(2,1,1,1)>()
#define bggb shuffle4_ro<_MM_SHUFFLE(2,1,1,2)>()
#define aggb shuffle4_ro<_MM_SHUFFLE(2,1,1,3)>()
#define rbgb shuffle4_ro<_MM_SHUFFLE(2,1,2,0)>()
#define gbgb shuffle4_ro<_MM_SHUFFLE(2,1,2,1)>()
#define bbgb shuffle4_ro<_MM_SHUFFLE(2,1,2,2)>()
#define abgb shuffle4_ro<_MM_SHUFFLE(2,1,2,3)>()
#define gagb shuffle4_ro<_MM_SHUFFLE(2,1,3,1)>()
#define bagb shuffle4_ro<_MM_SHUFFLE(2,1,3,2)>()
#define aagb shuffle4_ro<_MM_SHUFFLE(2,1,3,3)>()
#define rrbb shuffle4_ro<_MM_SHUFFLE(2,2,0,0)>()
#define grbb shuffle4_ro<_MM_SHUFFLE(2,2,0,1)>()
#define brbb shuffle4_ro<_MM_SHUFFLE(2,2,0,2)>()
#define arbb shuffle4_ro<_MM_SHUFFLE(2,2,0,3)>()
#define rgbb shuffle4_ro<_MM_SHUFFLE(2,2,1,0)>()
#define ggbb shuffle4_ro<_MM_SHUFFLE(2,2,1,1)>()
#define bgbb shuffle4_ro<_MM_SHUFFLE(2,2,1,2)>()
#define agbb shuffle4_ro<_MM_SHUFFLE(2,2,1,3)>()
#define rbbb shuffle4_ro<_MM_SHUFFLE(2,2,2,0)>()
#define gbbb shuffle4_ro<_MM_SHUFFLE(2,2,2,1)>()
#define bbbb shuffle4_ro<_MM_SHUFFLE(2,2,2,2)>()
#define abbb shuffle4_ro<_MM_SHUFFLE(2,2,2,3)>()
#define rabb shuffle4_ro<_MM_SHUFFLE(2,2,3,0)>()
#define gabb shuffle4_ro<_MM_SHUFFLE(2,2,3,1)>()
#define babb shuffle4_ro<_MM_SHUFFLE(2,2,3,2)>()
#define aabb shuffle4_ro<_MM_SHUFFLE(2,2,3,3)>()
#define rrab shuffle4_ro<_MM_SHUFFLE(2,3,0,0)>()
#define brab shuffle4_ro<_MM_SHUFFLE(2,3,0,2)>()
#define arab shuffle4_ro<_MM_SHUFFLE(2,3,0,3)>()
#define ggab shuffle4_ro<_MM_SHUFFLE(2,3,1,1)>()
#define bgab shuffle4_ro<_MM_SHUFFLE(2,3,1,2)>()
#define agab shuffle4_ro<_MM_SHUFFLE(2,3,1,3)>()
#define rbab shuffle4_ro<_MM_SHUFFLE(2,3,2,0)>()
#define gbab shuffle4_ro<_MM_SHUFFLE(2,3,2,1)>()
#define bbab shuffle4_ro<_MM_SHUFFLE(2,3,2,2)>()
#define abab shuffle4_ro<_MM_SHUFFLE(2,3,2,3)>()
#define raab shuffle4_ro<_MM_SHUFFLE(2,3,3,0)>()
#define gaab shuffle4_ro<_MM_SHUFFLE(2,3,3,1)>()
#define baab shuffle4_ro<_MM_SHUFFLE(2,3,3,2)>()
#define aaab shuffle4_ro<_MM_SHUFFLE(2,3,3,3)>()
#define rrra shuffle4_ro<_MM_SHUFFLE(3,0,0,0)>()
#define grra shuffle4_ro<_MM_SHUFFLE(3,0,0,1)>()
#define brra shuffle4_ro<_MM_SHUFFLE(3,0,0,2)>()
#define arra shuffle4_ro<_MM_SHUFFLE(3,0,0,3)>()
#define rgra shuffle4_ro<_MM_SHUFFLE(3,0,1,0)>()
#define ggra shuffle4_ro<_MM_SHUFFLE(3,0,1,1)>()
#define agra shuffle4_ro<_MM_SHUFFLE(3,0,1,3)>()
#define rbra shuffle4_ro<_MM_SHUFFLE(3,0,2,0)>()
#define bbra shuffle4_ro<_MM_SHUFFLE(3,0,2,2)>()
#define abra shuffle4_ro<_MM_SHUFFLE(3,0,2,3)>()
#define rara shuffle4_ro<_MM_SHUFFLE(3,0,3,0)>()
#define gara shuffle4_ro<_MM_SHUFFLE(3,0,3,1)>()
#define bara shuffle4_ro<_MM_SHUFFLE(3,0,3,2)>()
#define aara shuffle4_ro<_MM_SHUFFLE(3,0,3,3)>()
#define rrga shuffle4_ro<_MM_SHUFFLE(3,1,0,0)>()
#define grga shuffle4_ro<_MM_SHUFFLE(3,1,0,1)>()
#define arga shuffle4_ro<_MM_SHUFFLE(3,1,0,3)>()
#define rgga shuffle4_ro<_MM_SHUFFLE(3,1,1,0)>()
#define ggga shuffle4_ro<_MM_SHUFFLE(3,1,1,1)>()
#define bgga shuffle4_ro<_MM_SHUFFLE(3,1,1,2)>()
#define agga shuffle4_ro<_MM_SHUFFLE(3,1,1,3)>()
#define gbga shuffle4_ro<_MM_SHUFFLE(3,1,2,1)>()
#define bbga shuffle4_ro<_MM_SHUFFLE(3,1,2,2)>()
#define abga shuffle4_ro<_MM_SHUFFLE(3,1,2,3)>()
#define raga shuffle4_ro<_MM_SHUFFLE(3,1,3,0)>()
#define gaga shuffle4_ro<_MM_SHUFFLE(3,1,3,1)>()
#define baga shuffle4_ro<_MM_SHUFFLE(3,1,3,2)>()
#define aaga shuffle4_ro<_MM_SHUFFLE(3,1,3,3)>()
#define rrba shuffle4_ro<_MM_SHUFFLE(3,2,0,0)>()
#define brba shuffle4_ro<_MM_SHUFFLE(3,2,0,2)>()
#define arba shuffle4_ro<_MM_SHUFFLE(3,2,0,3)>()
#define ggba shuffle4_ro<_MM_SHUFFLE(3,2,1,1)>()
#define bgba shuffle4_ro<_MM_SHUFFLE(3,2,1,2)>()
#define agba shuffle4_ro<_MM_SHUFFLE(3,2,1,3)>()
#define rbba shuffle4_ro<_MM_SHUFFLE(3,2,2,0)>()
#define gbba shuffle4_ro<_MM_SHUFFLE(3,2,2,1)>()
#define bbba shuffle4_ro<_MM_SHUFFLE(3,2,2,2)>()
#define abba shuffle4_ro<_MM_SHUFFLE(3,2,2,3)>()
#define raba shuffle4_ro<_MM_SHUFFLE(3,2,3,0)>()
#define gaba shuffle4_ro<_MM_SHUFFLE(3,2,3,1)>()
#define baba shuffle4_ro<_MM_SHUFFLE(3,2,3,2)>()
#define aaba shuffle4_ro<_MM_SHUFFLE(3,2,3,3)>()
#define rraa shuffle4_ro<_MM_SHUFFLE(3,3,0,0)>()
#define graa shuffle4_ro<_MM_SHUFFLE(3,3,0,1)>()
#define braa shuffle4_ro<_MM_SHUFFLE(3,3,0,2)>()
#define araa shuffle4_ro<_MM_SHUFFLE(3,3,0,3)>()
#define rgaa shuffle4_ro<_MM_SHUFFLE(3,3,1,0)>()
#define ggaa shuffle4_ro<_MM_SHUFFLE(3,3,1,1)>()
#define bgaa shuffle4_ro<_MM_SHUFFLE(3,3,1,2)>()
#define agaa shuffle4_ro<_MM_SHUFFLE(3,3,1,3)>()
#define rbaa shuffle4_ro<_MM_SHUFFLE(3,3,2,0)>()
#define gbaa shuffle4_ro<_MM_SHUFFLE(3,3,2,1)>()
#define bbaa shuffle4_ro<_MM_SHUFFLE(3,3,2,2)>()
#define abaa shuffle4_ro<_MM_SHUFFLE(3,3,2,3)>()
#define raaa shuffle4_ro<_MM_SHUFFLE(3,3,3,0)>()
#define gaaa shuffle4_ro<_MM_SHUFFLE(3,3,3,1)>()
#define baaa shuffle4_ro<_MM_SHUFFLE(3,3,3,2)>()
#define aaaa shuffle4_ro<_MM_SHUFFLE(3,3,3,3)>()

// -------------------------------------------------------------------------- //

#define qpts shuffle4_rw<_MM_SHUFFLE(0,1,2,3)>()
#define pqts shuffle4_rw<_MM_SHUFFLE(0,1,3,2)>()
#define qtps shuffle4_rw<_MM_SHUFFLE(0,2,1,3)>()
#define tqps shuffle4_rw<_MM_SHUFFLE(0,2,3,1)>()
#define ptqs shuffle4_rw<_MM_SHUFFLE(0,3,1,2)>()
#define tpqs shuffle4_rw<_MM_SHUFFLE(0,3,2,1)>()
#define qpst shuffle4_rw<_MM_SHUFFLE(1,0,2,3)>()
#define pqst shuffle4_rw<_MM_SHUFFLE(1,0,3,2)>()
#define qspt shuffle4_rw<_MM_SHUFFLE(1,2,0,3)>()
#define sqpt shuffle4_rw<_MM_SHUFFLE(1,2,3,0)>()
#define psqt shuffle4_rw<_MM_SHUFFLE(1,3,0,2)>()
#define spqt shuffle4_rw<_MM_SHUFFLE(1,3,2,0)>()
#define qtsp shuffle4_rw<_MM_SHUFFLE(2,0,1,3)>()
#define tqsp shuffle4_rw<_MM_SHUFFLE(2,0,3,1)>()
#define qstp shuffle4_rw<_MM_SHUFFLE(2,1,0,3)>()
#define sqtp shuffle4_rw<_MM_SHUFFLE(2,1,3,0)>()
#define tsqp shuffle4_rw<_MM_SHUFFLE(2,3,0,1)>()
#define stqp shuffle4_rw<_MM_SHUFFLE(2,3,1,0)>()
#define ptsq shuffle4_rw<_MM_SHUFFLE(3,0,1,2)>()
#define tpsq shuffle4_rw<_MM_SHUFFLE(3,0,2,1)>()
#define pstq shuffle4_rw<_MM_SHUFFLE(3,1,0,2)>()
#define sptq shuffle4_rw<_MM_SHUFFLE(3,1,2,0)>()
#define tspq shuffle4_rw<_MM_SHUFFLE(3,2,0,1)>()
#define stpq shuffle4_rw<_MM_SHUFFLE(3,2,1,0)>()

#define ssss shuffle4_ro<_MM_SHUFFLE(0,0,0,0)>()
#define tsss shuffle4_ro<_MM_SHUFFLE(0,0,0,1)>()
#define psss shuffle4_ro<_MM_SHUFFLE(0,0,0,2)>()
#define qsss shuffle4_ro<_MM_SHUFFLE(0,0,0,3)>()
#define stss shuffle4_ro<_MM_SHUFFLE(0,0,1,0)>()
#define ttss shuffle4_ro<_MM_SHUFFLE(0,0,1,1)>()
#define ptss shuffle4_ro<_MM_SHUFFLE(0,0,1,2)>()
#define qtss shuffle4_ro<_MM_SHUFFLE(0,0,1,3)>()
#define spss shuffle4_ro<_MM_SHUFFLE(0,0,2,0)>()
#define tpss shuffle4_ro<_MM_SHUFFLE(0,0,2,1)>()
#define ppss shuffle4_ro<_MM_SHUFFLE(0,0,2,2)>()
#define qpss shuffle4_ro<_MM_SHUFFLE(0,0,2,3)>()
#define sqss shuffle4_ro<_MM_SHUFFLE(0,0,3,0)>()
#define tqss shuffle4_ro<_MM_SHUFFLE(0,0,3,1)>()
#define pqss shuffle4_ro<_MM_SHUFFLE(0,0,3,2)>()
#define qqss shuffle4_ro<_MM_SHUFFLE(0,0,3,3)>()
#define ssts shuffle4_ro<_MM_SHUFFLE(0,1,0,0)>()
#define tsts shuffle4_ro<_MM_SHUFFLE(0,1,0,1)>()
#define psts shuffle4_ro<_MM_SHUFFLE(0,1,0,2)>()
#define qsts shuffle4_ro<_MM_SHUFFLE(0,1,0,3)>()
#define stts shuffle4_ro<_MM_SHUFFLE(0,1,1,0)>()
#define ttts shuffle4_ro<_MM_SHUFFLE(0,1,1,1)>()
#define ptts shuffle4_ro<_MM_SHUFFLE(0,1,1,2)>()
#define qtts shuffle4_ro<_MM_SHUFFLE(0,1,1,3)>()
#define spts shuffle4_ro<_MM_SHUFFLE(0,1,2,0)>()
#define tpts shuffle4_ro<_MM_SHUFFLE(0,1,2,1)>()
#define ppts shuffle4_ro<_MM_SHUFFLE(0,1,2,2)>()
#define sqts shuffle4_ro<_MM_SHUFFLE(0,1,3,0)>()
#define tqts shuffle4_ro<_MM_SHUFFLE(0,1,3,1)>()
#define qqts shuffle4_ro<_MM_SHUFFLE(0,1,3,3)>()
#define ssps shuffle4_ro<_MM_SHUFFLE(0,2,0,0)>()
#define tsps shuffle4_ro<_MM_SHUFFLE(0,2,0,1)>()
#define psps shuffle4_ro<_MM_SHUFFLE(0,2,0,2)>()
#define qsps shuffle4_ro<_MM_SHUFFLE(0,2,0,3)>()
#define stps shuffle4_ro<_MM_SHUFFLE(0,2,1,0)>()
#define ttps shuffle4_ro<_MM_SHUFFLE(0,2,1,1)>()
#define ptps shuffle4_ro<_MM_SHUFFLE(0,2,1,2)>()
#define spps shuffle4_ro<_MM_SHUFFLE(0,2,2,0)>()
#define tpps shuffle4_ro<_MM_SHUFFLE(0,2,2,1)>()
#define ppps shuffle4_ro<_MM_SHUFFLE(0,2,2,2)>()
#define qpps shuffle4_ro<_MM_SHUFFLE(0,2,2,3)>()
#define sqps shuffle4_ro<_MM_SHUFFLE(0,2,3,0)>()
#define pqps shuffle4_ro<_MM_SHUFFLE(0,2,3,2)>()
#define qqps shuffle4_ro<_MM_SHUFFLE(0,2,3,3)>()
#define ssqs shuffle4_ro<_MM_SHUFFLE(0,3,0,0)>()
#define tsqs shuffle4_ro<_MM_SHUFFLE(0,3,0,1)>()
#define psqs shuffle4_ro<_MM_SHUFFLE(0,3,0,2)>()
#define qsqs shuffle4_ro<_MM_SHUFFLE(0,3,0,3)>()
#define stqs shuffle4_ro<_MM_SHUFFLE(0,3,1,0)>()
#define ttqs shuffle4_ro<_MM_SHUFFLE(0,3,1,1)>()
#define qtqs shuffle4_ro<_MM_SHUFFLE(0,3,1,3)>()
#define spqs shuffle4_ro<_MM_SHUFFLE(0,3,2,0)>()
#define ppqs shuffle4_ro<_MM_SHUFFLE(0,3,2,2)>()
#define qpqs shuffle4_ro<_MM_SHUFFLE(0,3,2,3)>()
#define sqqs shuffle4_ro<_MM_SHUFFLE(0,3,3,0)>()
#define tqqs shuffle4_ro<_MM_SHUFFLE(0,3,3,1)>()
#define pqqs shuffle4_ro<_MM_SHUFFLE(0,3,3,2)>()
#define qqqs shuffle4_ro<_MM_SHUFFLE(0,3,3,3)>()
#define ssst shuffle4_ro<_MM_SHUFFLE(1,0,0,0)>()
#define tsst shuffle4_ro<_MM_SHUFFLE(1,0,0,1)>()
#define psst shuffle4_ro<_MM_SHUFFLE(1,0,0,2)>()
#define qsst shuffle4_ro<_MM_SHUFFLE(1,0,0,3)>()
#define stst shuffle4_ro<_MM_SHUFFLE(1,0,1,0)>()
#define ttst shuffle4_ro<_MM_SHUFFLE(1,0,1,1)>()
#define ptst shuffle4_ro<_MM_SHUFFLE(1,0,1,2)>()
#define qtst shuffle4_ro<_MM_SHUFFLE(1,0,1,3)>()
#define spst shuffle4_ro<_MM_SHUFFLE(1,0,2,0)>()
#define tpst shuffle4_ro<_MM_SHUFFLE(1,0,2,1)>()
#define ppst shuffle4_ro<_MM_SHUFFLE(1,0,2,2)>()
#define sqst shuffle4_ro<_MM_SHUFFLE(1,0,3,0)>()
#define tqst shuffle4_ro<_MM_SHUFFLE(1,0,3,1)>()
#define qqst shuffle4_ro<_MM_SHUFFLE(1,0,3,3)>()
#define sstt shuffle4_ro<_MM_SHUFFLE(1,1,0,0)>()
#define tstt shuffle4_ro<_MM_SHUFFLE(1,1,0,1)>()
#define pstt shuffle4_ro<_MM_SHUFFLE(1,1,0,2)>()
#define qstt shuffle4_ro<_MM_SHUFFLE(1,1,0,3)>()
#define sttt shuffle4_ro<_MM_SHUFFLE(1,1,1,0)>()
#define tttt shuffle4_ro<_MM_SHUFFLE(1,1,1,1)>()
#define pttt shuffle4_ro<_MM_SHUFFLE(1,1,1,2)>()
#define qttt shuffle4_ro<_MM_SHUFFLE(1,1,1,3)>()
#define sptt shuffle4_ro<_MM_SHUFFLE(1,1,2,0)>()
#define tptt shuffle4_ro<_MM_SHUFFLE(1,1,2,1)>()
#define pptt shuffle4_ro<_MM_SHUFFLE(1,1,2,2)>()
#define qptt shuffle4_ro<_MM_SHUFFLE(1,1,2,3)>()
#define sqtt shuffle4_ro<_MM_SHUFFLE(1,1,3,0)>()
#define tqtt shuffle4_ro<_MM_SHUFFLE(1,1,3,1)>()
#define pqtt shuffle4_ro<_MM_SHUFFLE(1,1,3,2)>()
#define qqtt shuffle4_ro<_MM_SHUFFLE(1,1,3,3)>()
#define sspt shuffle4_ro<_MM_SHUFFLE(1,2,0,0)>()
#define tspt shuffle4_ro<_MM_SHUFFLE(1,2,0,1)>()
#define pspt shuffle4_ro<_MM_SHUFFLE(1,2,0,2)>()
#define stpt shuffle4_ro<_MM_SHUFFLE(1,2,1,0)>()
#define ttpt shuffle4_ro<_MM_SHUFFLE(1,2,1,1)>()
#define ptpt shuffle4_ro<_MM_SHUFFLE(1,2,1,2)>()
#define qtpt shuffle4_ro<_MM_SHUFFLE(1,2,1,3)>()
#define sppt shuffle4_ro<_MM_SHUFFLE(1,2,2,0)>()
#define tppt shuffle4_ro<_MM_SHUFFLE(1,2,2,1)>()
#define pppt shuffle4_ro<_MM_SHUFFLE(1,2,2,2)>()
#define qppt shuffle4_ro<_MM_SHUFFLE(1,2,2,3)>()
#define tqpt shuffle4_ro<_MM_SHUFFLE(1,2,3,1)>()
#define pqpt shuffle4_ro<_MM_SHUFFLE(1,2,3,2)>()
#define qqpt shuffle4_ro<_MM_SHUFFLE(1,2,3,3)>()
#define ssqt shuffle4_ro<_MM_SHUFFLE(1,3,0,0)>()
#define tsqt shuffle4_ro<_MM_SHUFFLE(1,3,0,1)>()
#define qsqt shuffle4_ro<_MM_SHUFFLE(1,3,0,3)>()
#define stqt shuffle4_ro<_MM_SHUFFLE(1,3,1,0)>()
#define ttqt shuffle4_ro<_MM_SHUFFLE(1,3,1,1)>()
#define ptqt shuffle4_ro<_MM_SHUFFLE(1,3,1,2)>()
#define qtqt shuffle4_ro<_MM_SHUFFLE(1,3,1,3)>()
#define tpqt shuffle4_ro<_MM_SHUFFLE(1,3,2,1)>()
#define ppqt shuffle4_ro<_MM_SHUFFLE(1,3,2,2)>()
#define qpqt shuffle4_ro<_MM_SHUFFLE(1,3,2,3)>()
#define sqqt shuffle4_ro<_MM_SHUFFLE(1,3,3,0)>()
#define tqqt shuffle4_ro<_MM_SHUFFLE(1,3,3,1)>()
#define pqqt shuffle4_ro<_MM_SHUFFLE(1,3,3,2)>()
#define qqqt shuffle4_ro<_MM_SHUFFLE(1,3,3,3)>()
#define sssp shuffle4_ro<_MM_SHUFFLE(2,0,0,0)>()
#define tssp shuffle4_ro<_MM_SHUFFLE(2,0,0,1)>()
#define pssp shuffle4_ro<_MM_SHUFFLE(2,0,0,2)>()
#define qssp shuffle4_ro<_MM_SHUFFLE(2,0,0,3)>()
#define stsp shuffle4_ro<_MM_SHUFFLE(2,0,1,0)>()
#define ttsp shuffle4_ro<_MM_SHUFFLE(2,0,1,1)>()
#define ptsp shuffle4_ro<_MM_SHUFFLE(2,0,1,2)>()
#define spsp shuffle4_ro<_MM_SHUFFLE(2,0,2,0)>()
#define tpsp shuffle4_ro<_MM_SHUFFLE(2,0,2,1)>()
#define ppsp shuffle4_ro<_MM_SHUFFLE(2,0,2,2)>()
#define qpsp shuffle4_ro<_MM_SHUFFLE(2,0,2,3)>()
#define sqsp shuffle4_ro<_MM_SHUFFLE(2,0,3,0)>()
#define pqsp shuffle4_ro<_MM_SHUFFLE(2,0,3,2)>()
#define qqsp shuffle4_ro<_MM_SHUFFLE(2,0,3,3)>()
#define sstp shuffle4_ro<_MM_SHUFFLE(2,1,0,0)>()
#define tstp shuffle4_ro<_MM_SHUFFLE(2,1,0,1)>()
#define pstp shuffle4_ro<_MM_SHUFFLE(2,1,0,2)>()
#define sttp shuffle4_ro<_MM_SHUFFLE(2,1,1,0)>()
#define tttp shuffle4_ro<_MM_SHUFFLE(2,1,1,1)>()
#define pttp shuffle4_ro<_MM_SHUFFLE(2,1,1,2)>()
#define qttp shuffle4_ro<_MM_SHUFFLE(2,1,1,3)>()
#define sptp shuffle4_ro<_MM_SHUFFLE(2,1,2,0)>()
#define tptp shuffle4_ro<_MM_SHUFFLE(2,1,2,1)>()
#define pptp shuffle4_ro<_MM_SHUFFLE(2,1,2,2)>()
#define qptp shuffle4_ro<_MM_SHUFFLE(2,1,2,3)>()
#define tqtp shuffle4_ro<_MM_SHUFFLE(2,1,3,1)>()
#define pqtp shuffle4_ro<_MM_SHUFFLE(2,1,3,2)>()
#define qqtp shuffle4_ro<_MM_SHUFFLE(2,1,3,3)>()
#define sspp shuffle4_ro<_MM_SHUFFLE(2,2,0,0)>()
#define tspp shuffle4_ro<_MM_SHUFFLE(2,2,0,1)>()
#define pspp shuffle4_ro<_MM_SHUFFLE(2,2,0,2)>()
#define qspp shuffle4_ro<_MM_SHUFFLE(2,2,0,3)>()
#define stpp shuffle4_ro<_MM_SHUFFLE(2,2,1,0)>()
#define ttpp shuffle4_ro<_MM_SHUFFLE(2,2,1,1)>()
#define ptpp shuffle4_ro<_MM_SHUFFLE(2,2,1,2)>()
#define qtpp shuffle4_ro<_MM_SHUFFLE(2,2,1,3)>()
#define sppp shuffle4_ro<_MM_SHUFFLE(2,2,2,0)>()
#define tppp shuffle4_ro<_MM_SHUFFLE(2,2,2,1)>()
#define pppp shuffle4_ro<_MM_SHUFFLE(2,2,2,2)>()
#define qppp shuffle4_ro<_MM_SHUFFLE(2,2,2,3)>()
#define sqpp shuffle4_ro<_MM_SHUFFLE(2,2,3,0)>()
#define tqpp shuffle4_ro<_MM_SHUFFLE(2,2,3,1)>()
#define pqpp shuffle4_ro<_MM_SHUFFLE(2,2,3,2)>()
#define qqpp shuffle4_ro<_MM_SHUFFLE(2,2,3,3)>()
#define ssqp shuffle4_ro<_MM_SHUFFLE(2,3,0,0)>()
#define psqp shuffle4_ro<_MM_SHUFFLE(2,3,0,2)>()
#define qsqp shuffle4_ro<_MM_SHUFFLE(2,3,0,3)>()
#define ttqp shuffle4_ro<_MM_SHUFFLE(2,3,1,1)>()
#define ptqp shuffle4_ro<_MM_SHUFFLE(2,3,1,2)>()
#define qtqp shuffle4_ro<_MM_SHUFFLE(2,3,1,3)>()
#define spqp shuffle4_ro<_MM_SHUFFLE(2,3,2,0)>()
#define tpqp shuffle4_ro<_MM_SHUFFLE(2,3,2,1)>()
#define ppqp shuffle4_ro<_MM_SHUFFLE(2,3,2,2)>()
#define qpqp shuffle4_ro<_MM_SHUFFLE(2,3,2,3)>()
#define sqqp shuffle4_ro<_MM_SHUFFLE(2,3,3,0)>()
#define tqqp shuffle4_ro<_MM_SHUFFLE(2,3,3,1)>()
#define pqqp shuffle4_ro<_MM_SHUFFLE(2,3,3,2)>()
#define qqqp shuffle4_ro<_MM_SHUFFLE(2,3,3,3)>()
#define sssq shuffle4_ro<_MM_SHUFFLE(3,0,0,0)>()
#define tssq shuffle4_ro<_MM_SHUFFLE(3,0,0,1)>()
#define pssq shuffle4_ro<_MM_SHUFFLE(3,0,0,2)>()
#define qssq shuffle4_ro<_MM_SHUFFLE(3,0,0,3)>()
#define stsq shuffle4_ro<_MM_SHUFFLE(3,0,1,0)>()
#define ttsq shuffle4_ro<_MM_SHUFFLE(3,0,1,1)>()
#define qtsq shuffle4_ro<_MM_SHUFFLE(3,0,1,3)>()
#define spsq shuffle4_ro<_MM_SHUFFLE(3,0,2,0)>()
#define ppsq shuffle4_ro<_MM_SHUFFLE(3,0,2,2)>()
#define qpsq shuffle4_ro<_MM_SHUFFLE(3,0,2,3)>()
#define sqsq shuffle4_ro<_MM_SHUFFLE(3,0,3,0)>()
#define tqsq shuffle4_ro<_MM_SHUFFLE(3,0,3,1)>()
#define pqsq shuffle4_ro<_MM_SHUFFLE(3,0,3,2)>()
#define qqsq shuffle4_ro<_MM_SHUFFLE(3,0,3,3)>()
#define sstq shuffle4_ro<_MM_SHUFFLE(3,1,0,0)>()
#define tstq shuffle4_ro<_MM_SHUFFLE(3,1,0,1)>()
#define qstq shuffle4_ro<_MM_SHUFFLE(3,1,0,3)>()
#define sttq shuffle4_ro<_MM_SHUFFLE(3,1,1,0)>()
#define tttq shuffle4_ro<_MM_SHUFFLE(3,1,1,1)>()
#define pttq shuffle4_ro<_MM_SHUFFLE(3,1,1,2)>()
#define qttq shuffle4_ro<_MM_SHUFFLE(3,1,1,3)>()
#define tptq shuffle4_ro<_MM_SHUFFLE(3,1,2,1)>()
#define pptq shuffle4_ro<_MM_SHUFFLE(3,1,2,2)>()
#define qptq shuffle4_ro<_MM_SHUFFLE(3,1,2,3)>()
#define sqtq shuffle4_ro<_MM_SHUFFLE(3,1,3,0)>()
#define tqtq shuffle4_ro<_MM_SHUFFLE(3,1,3,1)>()
#define pqtq shuffle4_ro<_MM_SHUFFLE(3,1,3,2)>()
#define qqtq shuffle4_ro<_MM_SHUFFLE(3,1,3,3)>()
#define sspq shuffle4_ro<_MM_SHUFFLE(3,2,0,0)>()
#define pspq shuffle4_ro<_MM_SHUFFLE(3,2,0,2)>()
#define qspq shuffle4_ro<_MM_SHUFFLE(3,2,0,3)>()
#define ttpq shuffle4_ro<_MM_SHUFFLE(3,2,1,1)>()
#define ptpq shuffle4_ro<_MM_SHUFFLE(3,2,1,2)>()
#define qtpq shuffle4_ro<_MM_SHUFFLE(3,2,1,3)>()
#define sppq shuffle4_ro<_MM_SHUFFLE(3,2,2,0)>()
#define tppq shuffle4_ro<_MM_SHUFFLE(3,2,2,1)>()
#define pppq shuffle4_ro<_MM_SHUFFLE(3,2,2,2)>()
#define qppq shuffle4_ro<_MM_SHUFFLE(3,2,2,3)>()
#define sqpq shuffle4_ro<_MM_SHUFFLE(3,2,3,0)>()
#define tqpq shuffle4_ro<_MM_SHUFFLE(3,2,3,1)>()
#define pqpq shuffle4_ro<_MM_SHUFFLE(3,2,3,2)>()
#define qqpq shuffle4_ro<_MM_SHUFFLE(3,2,3,3)>()
#define ssqq shuffle4_ro<_MM_SHUFFLE(3,3,0,0)>()
#define tsqq shuffle4_ro<_MM_SHUFFLE(3,3,0,1)>()
#define psqq shuffle4_ro<_MM_SHUFFLE(3,3,0,2)>()
#define qsqq shuffle4_ro<_MM_SHUFFLE(3,3,0,3)>()
#define stqq shuffle4_ro<_MM_SHUFFLE(3,3,1,0)>()
#define ttqq shuffle4_ro<_MM_SHUFFLE(3,3,1,1)>()
#define ptqq shuffle4_ro<_MM_SHUFFLE(3,3,1,2)>()
#define qtqq shuffle4_ro<_MM_SHUFFLE(3,3,1,3)>()
#define spqq shuffle4_ro<_MM_SHUFFLE(3,3,2,0)>()
#define tpqq shuffle4_ro<_MM_SHUFFLE(3,3,2,1)>()
#define ppqq shuffle4_ro<_MM_SHUFFLE(3,3,2,2)>()
#define qpqq shuffle4_ro<_MM_SHUFFLE(3,3,2,3)>()
#define sqqq shuffle4_ro<_MM_SHUFFLE(3,3,3,0)>()
#define tqqq shuffle4_ro<_MM_SHUFFLE(3,3,3,1)>()
#define pqqq shuffle4_ro<_MM_SHUFFLE(3,3,3,2)>()
#define qqqq shuffle4_ro<_MM_SHUFFLE(3,3,3,3)>()

#endif
