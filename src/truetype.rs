use std::collections::HashMap;
use std::convert::Infallible;
use std::iter;
use std::ops::Deref;
use crate::opentype::post::parse_post;
use crate::{Encoder, Font, FontError, Glyph, GlyphId, HMetrics, IResultExt, Info, Name, Pen, Shape, R};
use crate::parsers::{iterator, parse};
use itertools::Itertools;
use pathfinder_geometry::transform2d::Matrix2x2F;
use pdf_encoding::Encoding;
use nom::{
    number::complete::{be_u8, be_i8, be_i16, be_u16},
    bytes::complete::take,
    sequence::tuple
};
use pathfinder_geometry::{vector::Vector2F, transform2d::Transform2F, rect::RectF};
use crate::opentype::{
    parse_tables, parse_head, parse_maxp, parse_loca,
    parse_hhea, parse_hmtx, parse_name, Hmtx, Tables,
    cmap::{CMap, parse_cmap},
    kern::{parse_kern},
    gpos::KernTable,
    os2::parse_os2,
};


#[derive(Clone)]
pub struct TrueTypeFont<E: Encoder> {
    glyphs: Vec<Glyph<E>>,
    pub cmap: Option<CMap>,
    encoding: Option<Encoding>,
    hmtx: Hmtx,
    units_per_em: u16,
    bbox: RectF,
    kern: KernTable,
    name: Name,
    info: Info,
    pub name_map: HashMap<String, u16>,
}

#[derive(Clone)]
pub struct GlyphStorage {
    pub loca: Vec<u32>,
    pub glyf: Vec<u8>,
}

impl<E: Encoder> TrueTypeFont<E> {
    pub fn parse(data: &[u8], encoder: &mut E) -> Result<Self, FontError> {
        let tables = parse_tables(data)?;
        TrueTypeFont::parse_glyf(tables, encoder)
    }
    pub fn parse_glyf(tables: Tables<impl Deref<Target=[u8]>>, encoder: &mut E) -> Result<Self, FontError> {
        let head = parse_head(expect!(tables.get(b"head"), "no head"))?;
        let maxp = parse_maxp(expect!(tables.get(b"maxp"), "no maxp"))?;
        let loca = parse_loca(expect!(tables.get(b"loca"), "no loca"), &head, &maxp)?;
        let hhea = parse_hhea(expect!(tables.get(b"hhea"), "no hhea"))?;
        let hmtx = parse_hmtx(expect!(tables.get(b"hmtx"), "no hmtx"), &hhea, &maxp).get()?;
        
        let glyphs = parse_glyphs(&loca, expect!(tables.get(b"glyf"), "no glyf"), &hmtx, encoder)?;
        
        TrueTypeFont::from_shapes_and_metrics(tables, glyphs, hmtx)
    }
    pub fn from_shapes_and_metrics(tables: Tables<impl Deref<Target=[u8]>>, glyphs: Vec<Glyph<E>>, hmtx: Hmtx) -> Result<TrueTypeFont<E>, FontError> {
        let head = parse_head(expect!(tables.get(b"head"), "no head"))?;
        let (cmap, encoding) = match tables.get(b"cmap").map(|data| parse_cmap(data)).transpose()? {
            Some((cmap, encoding)) => (Some(cmap), Some(encoding)),
            None => (None, None)
        };
        let name = tables.get(b"name").map(|data| parse_name(data)).transpose()?.unwrap_or_default();
        let os2 = tables.get(b"OS/2").map(|data| parse_os2(data)).transpose()?;

        let post = t!(tables.get(b"post").map(parse_post).transpose());
        let mut name_map = HashMap::new();
        if let Some(post) = post {
            name_map.extend(post.names.into_iter().enumerate().map(|(i, name)| (name.into(), i as u16)));
        }
        
        Ok(TrueTypeFont {
            cmap,
            encoding,
            hmtx,
            units_per_em: head.units_per_em,
            bbox: head.bbox(),
            kern: tables.get(b"kern").map(|data| parse_kern(data)).transpose()?.unwrap_or_default(),
            name,
            info: Info {
                weight: os2.map(|t| t.weight),
            },
            name_map,
            glyphs,
        })
    }
}


impl<E: Encoder + 'static> Font<E> for TrueTypeFont<E> {
    fn num_glyphs(&self) -> u32 {
        self.glyphs.len() as u32
    }
    fn font_matrix(&self) -> Transform2F {
        let scale = 1.0 / self.units_per_em as f32;
        Transform2F::from_scale(Vector2F::splat(scale.into()))
    }
    fn glyph(&self, id: GlyphId) -> Option<&Glyph<E>> {
        self.glyphs.get(id.0 as usize)
    }
    fn is_empty_glyph(&self, gid: GlyphId) -> bool {
        match self.glyphs.get(gid.0 as usize) {
            None => true,
            Some(Glyph { metrics, shape: Shape::Empty }) => true,
            _ => false
        }
    }
    fn gid_for_codepoint(&self, codepoint: u32) -> Option<GlyphId> {
        match self.cmap {
            Some(ref cmap) => cmap.get_codepoint(codepoint).map(GlyphId),
            None => None
        }
    }
    fn gid_for_unicode_codepoint(&self, codepoint: u32) -> Option<GlyphId> {
        match (self.cmap.as_ref(), self.encoding) {
            (Some(cmap), Some(Encoding::Unicode)) => cmap.get_codepoint(codepoint).map(GlyphId),
            (Some(cmap), _) => self.gid_for_codepoint(codepoint),
            _ => None
        }
    }
    fn encoding(&self) -> Option<Encoding> {
        self.encoding
    }
    fn bbox(&self) -> Option<RectF> {
        Some(self.bbox)
    }
    fn kerning(&self, left: GlyphId, right: GlyphId) -> f32 {
        self.kern.get(left.0 as u16, right.0 as u16).unwrap_or(0) as f32
    }
    fn name(&self) -> &Name {
        &self.name
    }
    fn info(&self) -> &Info {
        &self.info
    }
}

#[inline]
fn vec_i8(i: &[u8]) -> R<Vector2F> {
    let (i, x) = be_i8(i)?;
    let (i, y) = be_i8(i)?;
    Ok((i, Vector2F::new(x as f32, y as f32)))
}
#[inline]
fn vec_i16(i: &[u8]) -> R<Vector2F> {
    let (i, x) = be_i16(i)?;
    let (i, y) = be_i16(i)?;
    Ok((i, Vector2F::new(x as f32, y as f32)))
}
#[inline]
fn fraction_i16(i: &[u8]) -> R<f32> {
    let (i, s) = be_i16(i)?;
    Ok((i, s as f32 / 16384.0))
}

pub fn parse_glyphs<E: Encoder>(loca: &[u32], data: &[u8], hmtx: &Hmtx, encoder: &mut E) -> Result<Vec<Glyph<E>>, FontError> {
    let mut glyphs = Vec::with_capacity(loca.len() - 1);
    for (i, (start, end)) in loca.iter().cloned().tuple_windows().enumerate() {
        let slice = expect!(data.get(start as usize .. end as usize), "out of bounds");
        //debug!("gid {} : data[{} .. {}]", i, start, end);
        let shape = parse_glyph_shape(slice, encoder)?;
        let metrics = hmtx.metrics_for_gid(i as u16);
        glyphs.push(Glyph { shape, metrics });
    }
    Ok(glyphs)
}
// the following code is borrowed from stb-truetype and modified heavily

fn parse_glyph_shape<E: Encoder>(data: &[u8], encoder: &mut E) -> Result<Shape<E>, FontError> {
    if data.len() == 0 {
        return Ok(Shape::Empty);
    }
    let (i, number_of_contours) = be_i16(data)?;
    
    let (i, _) = take(8usize)(i)?;
    //debug!("n_contours: {}", number_of_contours);
    match number_of_contours {
        0 => Ok(Shape::Empty),
        n if n >= 0 => glyph_shape_positive_contours(i, number_of_contours as usize, encoder),
        -1 => compound(i).get(),
        n => error!("Contour format {} not supported.", n)
    }
}

pub fn compound<E: Encoder>(mut input: &[u8]) -> R<Shape<E>> {
    // Compound shapes
    let mut parts = Vec::new();
    loop {
        let (flags, gidx) = parse(&mut input, tuple((be_u16, be_u16)))?;
        let mut transform = Transform2F::default();
        if flags & 2 != 0 {
            // XY values
            if flags & 1 != 0 {
                // shorts
                transform.vector = parse(&mut input, vec_i16)?
            } else {
                transform.vector = parse(&mut input, vec_i8)?
            }
        } else {
            panic!("Matching points not supported.");
        };
        if flags & (1 << 3) != 0 {
            // WE_HAVE_A_SCALE
            let scale = parse(&mut input, fraction_i16)?;
            transform.matrix = Matrix2x2F::from_scale(Vector2F::splat(scale));
        } else if flags & (1 << 6) != 0 {
            // WE_HAVE_AN_X_AND_YSCALE
            let (sx, sy) = parse(&mut input, tuple((fraction_i16, fraction_i16)))?;
            let s = Vector2F::new(sx, sy);
            transform.matrix = Matrix2x2F::from_scale(s);
        } else if flags & (1 << 7) != 0 {
            // WE_HAVE_A_TWO_BY_TWO
            let (a, b, c, d) = parse(&mut input, tuple((fraction_i16, fraction_i16, fraction_i16, fraction_i16)))?;
            transform.matrix = Matrix2x2F::row_major(a, b, c, d);
        }

        // Get indexed glyph.
        parts.push((GlyphId(gidx as u32), transform));
        // More components ?
        if flags & 0x20 == 0 {
            break;
        }
    }
    Ok((input, Shape::Compound(parts)))
}


#[derive(Copy, Clone, Debug)]
struct FlagData {
    flags: u8,
    p: (i32, i32)
}
fn parse_coord(short: bool, same_or_pos: bool) -> impl Fn(&[u8]) -> R<i16> {
    move |i| match (short, same_or_pos) {
        (true, true) => {
            let (i, dx) = be_u8(i)?;
            Ok((i, dx as i16))
        }
        (true, false) => {
            let (i, dx) = be_u8(i)?;
            Ok((i, - (dx as i16)))
        }
        (false, false) => {
            let (i, dx) = be_i16(i)?;
            Ok((i, dx))
        }
        (false, true) => Ok((i, 0))
    }
}
fn mid(a: Vector2F, b: Vector2F) -> Vector2F {
    (a + b) * 0.5
}
fn glyph_shape_positive_contours<E: Encoder>(i: &[u8], number_of_contours: usize, encoder: &mut E) -> Result<Shape<E>, FontError> {
    let (i, point_indices) = take(2 * number_of_contours)(i)?;
    let (i, num_instructions) = be_u16(i)?;
    let (mut i, _instructions) = take(num_instructions)(i)?;
    
    // total number of points
    let n = 1 + be_u16(slice!(point_indices, 2 * number_of_contours - 2 ..)).get()? as usize;

    let mut flag_data = Vec::with_capacity(n);

    // first load flags
    while flag_data.len() < n {
        let flags = parse(&mut i, be_u8)?;
        let flag = FlagData { flags, p: (0, 0) };
        
        if flags & 8 != 0 {
            let flagcount = parse(&mut i, be_u8)?;
            let num = (n - flag_data.len()).min(flagcount as usize + 1);
            flag_data.extend(iter::repeat(flag).take(num));
        } else {
            flag_data.push(flag);
        }
    }
    require_eq!(flag_data.len(), n);
    
    // now load x coordinates
    let mut x_coord: i32 = 0;
    for &mut FlagData { flags, ref mut p } in flag_data.iter_mut() {
        x_coord += parse(&mut i, parse_coord(flags & 2 != 0, flags & 16 != 0))? as i32;
        p.0 = x_coord;
    }

    // now load y coordinates
    let mut y_coord: i32 = 0;
    for &mut FlagData { flags, ref mut p } in flag_data.iter_mut() {
        y_coord += parse(&mut i, parse_coord(flags & 4 != 0, flags & 32 != 0))? as i32;
        p.1 = y_coord;
    }

    let mut points = flag_data.iter().map(|&FlagData { flags, p }| 
        (flags & 1 != 0, Vector2F::new(p.0 as f32, p.1 as f32))
    );
    let mut start = 0;
    let (_, glyph) = encoder.encode_shape::<(), Infallible>(move |mut pen| {
        for end in iterator(point_indices, be_u16) {
            let n_points = end + 1 - start;
            start += n_points;
            
            contour((&mut points).take(n_points as usize), &mut pen);
        }
        Ok(())
    }).unwrap();
    
    Ok(Shape::Simple(glyph))
}

pub fn contour(points: impl Iterator<Item=(bool, Vector2F)>, pen: &mut impl Pen) {
    let mut points = points.peekable();
    
    let Some((start_on, p)) = points.next() else { return };
    let start_off = !start_on;
    let (s, sc) = if start_off {
        // if we start off with an off-curve point, then when we need to find a
        // point on the curve where we can start, and we
        // need to save some state for
        // when we wraparound.
        let sc = p;

        let Some(&(next_on, next_p)) = points.peek() else { return };

        let p = if !next_on {
            // next point is also a curve point, so interpolate an on-point curve
            mid(p, next_p)
        } else {
            // we're using point i+1 as the starting point, so skip it
            let _ = points.next();
            
            // otherwise just use the next point as our start point
            next_p
        };
        (p, Some(sc))
    } else {
        (p, None)
    };
    
    pen.move_to(s);
    
    let mut c = None;
    for (on_curve, p) in points {
        if !on_curve {
            // if it's a curve
            if let Some(c) = c {
                // two off-curve control points in a row means interpolate an on-curve
                // midpoint
                pen.quad_to(c, mid(c, p));
            }
            c = Some(p);
        } else {
            if let Some(c) = c.take() {
                pen.quad_to(c, p);
            } else {
                pen.line_to(p);
            }
        }
    }
    
    if let Some(sc) = sc {
        if let Some(c) = c {
            pen.quad_to(c, mid(c, sc));
        }
        pen.quad_to(sc, s);
    } else {
        if let Some(c) = c {
            pen.quad_to(c, s);
        } else {
            pen.line_to(s);
        }
    }

    pen.close();
}
