// Calculates bounding rectangle of a point set or retrieves already calculated
static Rect pointSetBoundingRect( const Mat& points )
{
    int npoints = points.checkVector(2);
    int depth = points.depth();
    CV_Assert(npoints >= 0 && (depth == CV_32F || depth == CV_32S));

    int  xmin = 0, ymin = 0, xmax = -1, ymax = -1, i;
    bool is_float = depth == CV_32F;

    if( npoints == 0 )
        return Rect();

#if CV_SIMD
    const int64_t* pts = points.ptr<int64_t>();

    if( !is_float )
    {
        v_int32 minval, maxval;
        minval = maxval = v_reinterpret_as_s32(vx_setall_s64(*pts)); //min[0]=pt.x, min[1]=pt.y, min[2]=pt.x, min[3]=pt.y
        for( i = 1; i <= npoints - v_int32::nlanes/2; i+= v_int32::nlanes/2 )
        {
            v_int32 ptXY2 = v_reinterpret_as_s32(vx_load(pts + i));
            minval = v_min(ptXY2, minval);
            maxval = v_max(ptXY2, maxval);
        }
        minval = v_min(v_reinterpret_as_s32(v_expand_low(v_reinterpret_as_u32(minval))), v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(minval))));
        maxval = v_max(v_reinterpret_as_s32(v_expand_low(v_reinterpret_as_u32(maxval))), v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(maxval))));
        if( i <= npoints - v_int32::nlanes/4 )
        {
            v_int32 ptXY = v_reinterpret_as_s32(v_expand_low(v_reinterpret_as_u32(vx_load_low(pts + i))));
            minval = v_min(ptXY, minval);
            maxval = v_max(ptXY, maxval);
            i += v_int64::nlanes/2;
        }
        for(int j = 16; j < CV_SIMD_WIDTH; j*=2)
        {
            minval = v_min(v_reinterpret_as_s32(v_expand_low(v_reinterpret_as_u32(minval))), v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(minval))));
            maxval = v_max(v_reinterpret_as_s32(v_expand_low(v_reinterpret_as_u32(maxval))), v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(maxval))));
        }
        xmin = minval.get0();
        xmax = maxval.get0();
        ymin = v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(minval))).get0();
        ymax = v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(maxval))).get0();
#if CV_SIMD_WIDTH > 16
        if( i < npoints )
        {
            v_int32x4 minval2, maxval2;
            minval2 = maxval2 = v_reinterpret_as_s32(v_expand_low(v_reinterpret_as_u32(v_load_low(pts + i))));
            for( i++; i < npoints; i++ )
            {
                v_int32x4 ptXY = v_reinterpret_as_s32(v_expand_low(v_reinterpret_as_u32(v_load_low(pts + i))));
                minval2 = v_min(ptXY, minval2);
                maxval2 = v_max(ptXY, maxval2);
            }
            xmin = min(xmin, minval2.get0());
            xmax = max(xmax, maxval2.get0());
            ymin = min(ymin, v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(minval2))).get0());
            ymax = max(ymax, v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(maxval2))).get0());
        }
#endif
    }
    else
    {
        v_float32 minval, maxval;
        minval = maxval = v_reinterpret_as_f32(vx_setall_s64(*pts)); //min[0]=pt.x, min[1]=pt.y, min[2]=pt.x, min[3]=pt.y
        for( i = 1; i <= npoints - v_float32::nlanes/2; i+= v_float32::nlanes/2 )
        {
            v_float32 ptXY2 = v_reinterpret_as_f32(vx_load(pts + i));
            minval = v_min(ptXY2, minval);
            maxval = v_max(ptXY2, maxval);
        }
        minval = v_min(v_reinterpret_as_f32(v_expand_low(v_reinterpret_as_u32(minval))), v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(minval))));
        maxval = v_max(v_reinterpret_as_f32(v_expand_low(v_reinterpret_as_u32(maxval))), v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(maxval))));
        if( i <= npoints - v_float32::nlanes/4 )
        {
            v_float32 ptXY = v_reinterpret_as_f32(v_expand_low(v_reinterpret_as_u32(vx_load_low(pts + i))));
            minval = v_min(ptXY, minval);
            maxval = v_max(ptXY, maxval);
            i += v_float32::nlanes/4;
        }
        for(int j = 16; j < CV_SIMD_WIDTH; j*=2)
        {
            minval = v_min(v_reinterpret_as_f32(v_expand_low(v_reinterpret_as_u32(minval))), v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(minval))));
            maxval = v_max(v_reinterpret_as_f32(v_expand_low(v_reinterpret_as_u32(maxval))), v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(maxval))));
        }
        xmin = cvFloor(minval.get0());
        xmax = cvFloor(maxval.get0());
        ymin = cvFloor(v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(minval))).get0());
        ymax = cvFloor(v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(maxval))).get0());
#if CV_SIMD_WIDTH > 16
        if( i < npoints )
        {
            v_float32x4 minval2, maxval2;
            minval2 = maxval2 = v_reinterpret_as_f32(v_expand_low(v_reinterpret_as_u32(v_load_low(pts + i))));
            for( i++; i < npoints; i++ )
            {
                v_float32x4 ptXY = v_reinterpret_as_f32(v_expand_low(v_reinterpret_as_u32(v_load_low(pts + i))));
                minval2 = v_min(ptXY, minval2);
                maxval2 = v_max(ptXY, maxval2);
            }
            xmin = min(xmin, cvFloor(minval2.get0()));
            xmax = max(xmax, cvFloor(maxval2.get0()));
            ymin = min(ymin, cvFloor(v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(minval2))).get0()));
            ymax = max(ymax, cvFloor(v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(maxval2))).get0()));
        }
#endif
    }
#else
    const Point* pts = points.ptr<Point>();
    Point pt = pts[0];

    if( !is_float )
    {
        xmin = xmax = pt.x;
        ymin = ymax = pt.y;

        for( i = 1; i < npoints; i++ )
        {
            pt = pts[i];

            if( xmin > pt.x )
                xmin = pt.x;

            if( xmax < pt.x )
                xmax = pt.x;

            if( ymin > pt.y )
                ymin = pt.y;

            if( ymax < pt.y )
                ymax = pt.y;
        }
    }
    else
    {
        Cv32suf v;
        // init values
        xmin = xmax = CV_TOGGLE_FLT(pt.x);
        ymin = ymax = CV_TOGGLE_FLT(pt.y);

        for( i = 1; i < npoints; i++ )
        {
            pt = pts[i];
            pt.x = CV_TOGGLE_FLT(pt.x);
            pt.y = CV_TOGGLE_FLT(pt.y);

            if( xmin > pt.x )
                xmin = pt.x;

            if( xmax < pt.x )
                xmax = pt.x;

            if( ymin > pt.y )
                ymin = pt.y;

            if( ymax < pt.y )
                ymax = pt.y;
        }

        v.i = CV_TOGGLE_FLT(xmin); xmin = cvFloor(v.f);
        v.i = CV_TOGGLE_FLT(ymin); ymin = cvFloor(v.f);
        // because right and bottom sides of the bounding rectangle are not inclusive
        // (note +1 in width and height calculation below), cvFloor is used here instead of cvCeil
        v.i = CV_TOGGLE_FLT(xmax); xmax = cvFloor(v.f);
        v.i = CV_TOGGLE_FLT(ymax); ymax = cvFloor(v.f);
    }
#endif

    return Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
}


static Rect maskBoundingRect( const Mat& img )
{
    CV_Assert( img.depth() <= CV_8S && img.channels() == 1 );

    Size size = img.size();
    int xmin = size.width, ymin = -1, xmax = -1, ymax = -1, i, j, k;

    for( i = 0; i < size.height; i++ )
    {
        const uchar* _ptr = img.ptr(i);
        const uchar* ptr = (const uchar*)alignPtr(_ptr, 4);//字节右对齐
        int have_nz = 0, k_min, offset = (int)(ptr - _ptr);
        j = 0;
        offset = MIN(offset, size.width);
        //先处理没有对齐的部分，尝试找到最小值
        for( ; j < offset; j++ )
            if( _ptr[j] )
            {
                have_nz = 1;
                break;
            }
        if( j < offset )
        {
            if( j < xmin )
                xmin = j;
            if( j > xmax )
                xmax = j;
        }
        if( offset < size.width )
        {
            //对齐后其他参数相应改变，从offset之后开始坐标j=0,1.2
            xmin -= offset;
            xmax -= offset;
            size.width -= offset;
            j = 0;
            //xmin可代表上一个循环最小的值，不一定需要遍历到末尾，类似循环展开
            for( ; j <= xmin - 4; j += 4 )
                if( *((int*)(ptr+j)) )
                    break;
            for( ; j < xmin; j++ )
                if( ptr[j] )
                {
                    xmin = j;
                    if( j > xmax )
                        xmax = j;
                    have_nz = 1;
                    break;
                }
            k_min = MAX(j-1, xmax);
            k = size.width - 1;
            //从右往左遍历，直到k+1==4n,索引为４的倍数，循环展开的剩余部分
            for( ; k > k_min && (k&3) != 3; k-- )
                if( ptr[k] )
                    break;
            if( k > k_min && (k&3) == 3 )
            {
                //每４*uchar转一个int*强制同时处理，实现加速判断非零值
                //注意这里只能找到一个４*uchar片段，若存在非零，具体ｋ值还需在这个片段内判断
                for( ; k > k_min+3; k -= 4 )
                    if( *((int*)(ptr+k-3)) )
                        break;
            }
            //1.找到４*uchar片段中具体非零的位置
            //2.若前面都没有找到非零，以下代码则是处理[xmax,alignSize(xmax,4)-1]之间剩余的部分
            // alignSize(xmax,4)-1即为4n+3
            for( ; k > k_min; k-- )
                if( ptr[k] )
                {
                    xmax = k;
                    have_nz = 1;
                    break;
                }
            if( !have_nz )
            {
                j &= ~3;//j-k=4*n向下取４倍数0,4,8,12,16...
                //[0,offset]与(xmax,width]两个片段都没有找到，尝试遍历[offset－delta,xmax]
                for( ; j <= k - 3; j += 4 )
                    if( *((int*)(ptr+j)) )
                        break;
                for( ; j <= k; j++ )
                    if( ptr[j] )
                    {
                        have_nz = 1;
                        break;
                    }
            }
            xmin += offset;
            xmax += offset;
            size.width += offset;
        }
        if( have_nz )
        {
            if( ymin < 0 )
                ymin = i;//只赋值一次，从上往下，只找第一次出现的y值
            ymax = i;
        }
    }

    if( xmin >= size.width )
        xmin = ymin = 0;
    return Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
}

}
