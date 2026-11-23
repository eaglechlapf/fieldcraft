//! Pictures in a field's help ([`fieldcraft_engine::schema::Field::picture`]): small diagrams of
//! what a setting does, drawn here in the theme's colours, so they follow light and dark and stay
//! sharp at any scale. They hold no words (the help text under them does), and no file: there is
//! no asset to keep or attribute.
//!
//! Each is drawn on an artboard of [`WIDTH`] × [`HEIGHT`] units, mapped onto the rect it is
//! given: a part in the theme's soft accent, the grid's cells in its dim line colour, what the
//! setting changes in its warm colour, and a stress peak in its error red.

use egui::{Color32, Painter, Pos2, Rect, Shape, Stroke, Ui, pos2, vec2};

/// The artboard, in units: a picture is drawn this many points across and high.
pub const WIDTH: f32 = 240.0;
pub const HEIGHT: f32 = 132.0;

/// The widest a picture is drawn in a help, in points.
pub const MOST_WIDTH: f32 = 300.0;

/// Every picture, by its name.
pub const NAMES: &[&str] = &["grid.cell_size", "grid.adapt", "grid.refine", "grid.gaps", "grid.tolerance", "grid.sampling"];

/// The colours of a picture.
struct Ink {
    part: Color32,
    outline: Color32,
    grid: Color32,
    accent: Color32,
    warm: Color32,
    peak: Color32,
    /// The tooltip's own fill: a hole or a gap in the part.
    background: Color32,
}

impl Ink {
    fn of(ui: &Ui) -> Ink {
        let t = crate::theme::Tokens::get(ui.visuals().dark_mode);
        Ink {
            part: t.accent_soft,
            outline: t.text,
            grid: t.text_dim.gamma_multiply(0.55),
            accent: t.accent,
            warm: t.warm,
            peak: t.error,
            background: solidcraft_theme::Tokens::get().dialog_body,
        }
    }
}

/// Artboard units to the screen: the picture fills `rect`, keeping its proportions.
struct Board {
    painter: Painter,
    origin: Pos2,
    scale: f32,
}

impl Board {
    fn p(&self, x: f32, y: f32) -> Pos2 {
        pos2(self.origin.x + x * self.scale, self.origin.y + y * self.scale)
    }

    fn r(&self, x0: f32, y0: f32, x1: f32, y1: f32) -> Rect {
        Rect::from_two_pos(self.p(x0, y0), self.p(x1, y1))
    }

    fn fill(&self, x0: f32, y0: f32, x1: f32, y1: f32, color: Color32) {
        self.painter.rect_filled(self.r(x0, y0, x1, y1), 0.0, color);
    }

    fn line(&self, a: (f32, f32), b: (f32, f32), width: f32, color: Color32) {
        self.painter.line_segment([self.p(a.0, a.1), self.p(b.0, b.1)], Stroke::new(width * self.scale.max(0.5), color));
    }

    fn outline(&self, points: &[(f32, f32)], width: f32, color: Color32) {
        let points: Vec<Pos2> = points.iter().map(|(x, y)| self.p(*x, *y)).collect();
        self.painter.add(Shape::closed_line(points, Stroke::new(width * self.scale.max(0.5), color)));
    }

    fn convex(&self, points: &[(f32, f32)], color: Color32) {
        let points: Vec<Pos2> = points.iter().map(|(x, y)| self.p(*x, *y)).collect();
        self.painter.add(Shape::convex_polygon(points, color, Stroke::NONE));
    }

    fn square(&self, x: f32, y: f32, side: f32, width: f32, color: Color32) {
        self.outline(&[(x, y), (x + side, y), (x + side, y + side), (x, y + side)], width, color);
    }

    /// A cell from (`x`, `y`) of `side` divided into four, `levels` times toward (`cx`, `cy`):
    /// every level divides the quarter that holds that point, and its neighbours around it.
    fn divide(&self, x: f32, y: f32, side: f32, levels: usize, toward: (f32, f32), color: Color32) {
        if levels == 0 || side < 2.0 {
            return;
        }
        let half = side / 2.0;
        self.line((x + half, y), (x + half, y + side), 1.1, color);
        self.line((x, y + half), (x + side, y + half), 1.1, color);
        for (qx, qy) in [(x, y), (x + half, y), (x, y + half), (x + half, y + half)] {
            // The quarter that holds the point (on its edge or corner too) is divided again.
            let near = |c: f32, lo: f32| c >= lo - 0.01 && c <= lo + half + 0.01;
            if near(toward.0, qx) && near(toward.1, qy) {
                self.divide(qx, qy, half, levels - 1, toward, color);
            }
        }
    }

    /// The grid's lines from (`x0`, `y0`) to (`x1`, `y1`), cells of `cell`.
    fn grid(&self, x0: f32, y0: f32, x1: f32, y1: f32, cell: f32, color: Color32) {
        let mut x = x0;
        while x <= x1 + 0.01 {
            self.line((x, y0), (x, y1), 0.7, color);
            x += cell;
        }
        let mut y = y0;
        while y <= y1 + 0.01 {
            self.line((x0, y), (x1, y), 0.7, color);
            y += cell;
        }
    }

    /// A dimension from `a` to `b` (horizontal or vertical): a line with ticks at its ends.
    fn dimension(&self, a: (f32, f32), b: (f32, f32), color: Color32) {
        self.line(a, b, 1.2, color);
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let length = (dx * dx + dy * dy).sqrt().max(1e-3);
        let (nx, ny) = (-dy / length * 4.0, dx / length * 4.0);
        for end in [a, b] {
            self.line((end.0 - nx, end.1 - ny), (end.0 + nx, end.1 + ny), 1.2, color);
        }
    }

    /// A stress peak at (`x`, `y`): rings that fade outward.
    fn peak(&self, x: f32, y: f32, color: Color32) {
        for (radius, alpha) in [(16.0, 0.18), (10.0, 0.32), (5.0, 0.6)] {
            self.painter.circle_filled(self.p(x, y), radius * self.scale, color.gamma_multiply(alpha));
        }
    }
}

/// The L-shaped part most pictures show: a foot and an upright.
const FOOT: [f32; 4] = [30.0, 80.0, 210.0, 112.0];
const UPRIGHT: [f32; 4] = [30.0, 20.0, 110.0, 80.0];
const L_OUTLINE: [(f32, f32); 6] = [(30.0, 20.0), (110.0, 20.0), (110.0, 80.0), (210.0, 80.0), (210.0, 112.0), (30.0, 112.0)];

fn l_part(b: &Board, ink: &Ink) {
    b.fill(FOOT[0], FOOT[1], FOOT[2], FOOT[3], ink.part);
    b.fill(UPRIGHT[0], UPRIGHT[1], UPRIGHT[2], UPRIGHT[3], ink.part);
}

/// Cell size: the part in a grid of cubic cells, one cell's side marked.
fn cell_size(b: &Board, ink: &Ink) {
    l_part(b, ink);
    b.grid(20.0, 10.0, 220.0, 120.0, 20.0, ink.grid);
    b.outline(&L_OUTLINE, 1.6, ink.outline);
    b.fill(140.0, 90.0, 160.0, 110.0, ink.warm.gamma_multiply(0.35));
    b.square(140.0, 90.0, 20.0, 1.8, ink.warm);
    b.dimension((140.0, 126.0), (160.0, 126.0), ink.warm);
}

/// Adaptive rounds: a close-up of the inside corner, where the stress peaks; round by round, the
/// cells of material there are divided again.
fn adapt(b: &Board, ink: &Ink) {
    // The upright left of x = 120, the foot below y = 80.
    b.fill(0.0, 0.0, 120.0, HEIGHT, ink.part);
    b.fill(120.0, 80.0, WIDTH, HEIGHT, ink.part);
    b.grid(0.0, 0.0, WIDTH, 160.0, 40.0, ink.grid);
    b.peak(120.0, 80.0, ink.peak);
    for (x, y) in [(80.0, 40.0), (80.0, 80.0), (120.0, 80.0)] {
        b.divide(x, y, 40.0, 3, (120.0, 80.0), ink.warm);
    }
    b.line((120.0, 0.0), (120.0, 80.0), 1.8, ink.outline);
    b.line((120.0, 80.0), (WIDTH, 80.0), 1.8, ink.outline);
}

/// Refinement levels: a plate with a hole; the cells the hole's edge crosses divided once.
fn refine(b: &Board, ink: &Ink) {
    let (cx, cy, r) = (120.0, 66.0, 24.0);
    b.fill(30.0, 20.0, 210.0, 112.0, ink.part);
    b.painter.circle_filled(b.p(cx, cy), r * b.scale, ink.background);
    b.grid(20.0, 10.0, 220.0, 120.0, 20.0, ink.grid);
    let mut y = 10.0;
    while y < 120.0 {
        let mut x = 20.0;
        while x < 220.0 {
            // The nearest and the farthest point of the cell from the hole's centre.
            let dx = if cx < x {
                x - cx
            } else if cx > x + 20.0 {
                cx - x - 20.0
            } else {
                0.0
            };
            let dy = if cy < y {
                y - cy
            } else if cy > y + 20.0 {
                cy - y - 20.0
            } else {
                0.0
            };
            let nearest = (dx * dx + dy * dy).sqrt();
            let fx = (cx - x).abs().max((cx - x - 20.0).abs());
            let fy = (cy - y).abs().max((cy - y - 20.0).abs());
            let farthest = (fx * fx + fy * fy).sqrt();
            if nearest < r && farthest > r {
                b.divide(x, y, 20.0, 1, (x + 10.0, y + 10.0), ink.warm);
            }
            x += 20.0;
        }
        y += 20.0;
    }
    b.outline(&[(30.0, 20.0), (210.0, 20.0), (210.0, 112.0), (30.0, 112.0)], 1.6, ink.outline);
    let ring: Vec<(f32, f32)> = (0..48).map(|i| (i as f32) / 48.0 * std::f32::consts::TAU).map(|a| (cx + r * a.cos(), cy + r * a.sin())).collect();
    b.outline(&ring, 1.6, ink.outline);
}

/// Narrow gaps: a close-up of a slit narrower than a cell. The two sides keep apart, and the
/// cells where the slit ends are divided.
fn gaps(b: &Board, ink: &Ink) {
    let (end, top, bottom) = (110.0, 60.0, 70.0);
    b.fill(0.0, 0.0, WIDTH, HEIGHT, ink.part);
    b.fill(end, top, WIDTH, bottom, ink.background);
    b.grid(0.0, 0.0, WIDTH, 160.0, 40.0, ink.grid);
    b.divide(80.0, 40.0, 40.0, 2, (end, 65.0), ink.warm);
    b.divide(120.0, 40.0, 40.0, 1, (140.0, 60.0), ink.warm);
    b.outline(&[(WIDTH, top), (end, top), (end, bottom), (WIDTH, bottom)], 1.8, ink.outline);
}

/// Chordal tolerance: a curved face and the flat triangles that stand for it (drawn far coarser
/// than they are); the tolerance is how far they may lie from it.
fn tolerance(b: &Board, ink: &Ink) {
    let (cx, cy, r) = (120.0, 170.0, 140.0);
    let arc = |s: f32| {
        let a = -std::f32::consts::FRAC_PI_2 + (s - 0.5) * 1.6;
        (cx + r * a.cos(), cy + r * a.sin())
    };
    let mut face: Vec<(f32, f32)> = (0..=48).map(|i| arc(i as f32 / 48.0)).collect();
    let (first, last) = (face[0], face[face.len() - 1]);
    face.push((last.0, HEIGHT));
    face.push((first.0, HEIGHT));
    b.convex(&face, ink.part);
    let curve: Vec<Pos2> = (0..=48).map(|i| arc(i as f32 / 48.0)).map(|(x, y)| b.p(x, y)).collect();
    b.painter.add(Shape::line(curve, Stroke::new(1.8 * b.scale, ink.outline)));
    let chords = [arc(0.0), arc(0.5), arc(1.0)];
    for pair in chords.windows(2) {
        b.line(pair[0], pair[1], 1.8, ink.accent);
    }
    for end in chords {
        b.painter.circle_filled(b.p(end.0, end.1), 3.0 * b.scale, ink.accent);
    }
    // The largest distance: from the middle of a chord to the face.
    let middle = ((chords[0].0 + chords[1].0) / 2.0, (chords[0].1 + chords[1].1) / 2.0);
    b.dimension(middle, arc(0.25), ink.warm);
}

/// Result sampling: the results are read at points of the surface no farther apart than the
/// sampling, closer where the cells are smaller.
fn sampling(b: &Board, ink: &Ink) {
    let (cx, cy, r) = (120.0, 236.0, 196.0);
    let arc = |t: f32| {
        let a = -std::f32::consts::FRAC_PI_2 + (t - 0.5) * 1.1;
        (cx + r * a.cos(), cy + r * a.sin())
    };
    let mut face: Vec<(f32, f32)> = (0..=40).map(|i| arc(i as f32 / 40.0)).collect();
    let (first, last) = (face[0], face[face.len() - 1]);
    face.push((last.0, 124.0));
    face.push((first.0, 124.0));
    b.convex(&face, ink.part);
    b.grid(20.0, 10.0, 220.0, 130.0, 30.0, ink.grid);
    let curve: Vec<Pos2> = (0..=40).map(|i| arc(i as f32 / 40.0)).map(|(x, y)| b.p(x, y)).collect();
    b.painter.add(Shape::line(curve, Stroke::new(1.6 * b.scale, ink.outline)));
    for i in 0..=12 {
        let (x, y) = arc(i as f32 / 12.0);
        b.painter.circle_filled(b.p(x, y), 2.6 * b.scale, ink.warm);
    }
    let (a, c) = (arc(7.0 / 12.0), arc(8.0 / 12.0));
    b.dimension((a.0, a.1 - 12.0), (c.0, c.1 - 12.0), ink.warm);
}

/// Draw the picture `name` in `rect`; false when there is none of that name.
pub fn paint(ui: &Ui, name: &str, rect: Rect) -> bool {
    let draw: fn(&Board, &Ink) = match name {
        "grid.cell_size" => cell_size,
        "grid.adapt" => adapt,
        "grid.refine" => refine,
        "grid.gaps" => gaps,
        "grid.tolerance" => tolerance,
        "grid.sampling" => sampling,
        _ => return false,
    };
    let scale = (rect.width() / WIDTH).min(rect.height() / HEIGHT).max(0.0);
    let size = vec2(WIDTH, HEIGHT) * scale;
    let origin = rect.center() - size / 2.0;
    let board = Board { painter: ui.painter().with_clip_rect(Rect::from_min_size(origin, size).intersect(ui.clip_rect())), origin, scale };
    draw(&board, &Ink::of(ui));
    true
}

/// Allocate room for the picture `name`, `width` across, and draw it; false when there is none.
pub fn show(ui: &mut Ui, name: &str, width: f32) -> bool {
    if !NAMES.contains(&name) {
        return false;
    }
    let width = width.max(1.0);
    let (rect, _) = ui.allocate_exact_size(vec2(width, width * HEIGHT / WIDTH), egui::Sense::hover());
    paint(ui, name, rect)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every picture draws, in both themes, within its rect, and something of it; an unknown
    /// name draws nothing.
    #[test]
    fn every_picture_draws_inside_its_room() {
        let ctx = egui::Context::default();
        for dark in [false, true] {
            ctx.set_visuals(if dark { egui::Visuals::dark() } else { egui::Visuals::light() });
            for name in NAMES {
                let mut room = Rect::NOTHING;
                let mut out = ctx.run_ui(egui::RawInput::default(), |ui| {
                    assert!(show(ui, name, 240.0), "{name}");
                    room = ui.min_rect();
                });
                out.textures_delta.clear();
                let shapes: Vec<_> = out.shapes.iter().filter(|s| s.clip_rect.is_positive()).collect();
                assert!(shapes.len() > 5, "{name}: {} shapes", shapes.len());
                for s in shapes {
                    assert!(room.expand(1.0).contains_rect(s.clip_rect.intersect(room.expand(1.0))), "{name}");
                    assert!(
                        s.clip_rect.width() <= room.width() + 1.0 && s.clip_rect.height() <= room.height() + 1.0,
                        "{name}: {:?} in {room:?}",
                        s.clip_rect
                    );
                }
            }
        }
        let mut drawn = true;
        let mut out = ctx.run_ui(egui::RawInput::default(), |ui| drawn = show(ui, "no.such.picture", 240.0));
        out.textures_delta.clear();
        assert!(!drawn);
    }

    /// Every picture a command's form names is one of [`NAMES`], and every one of those is named
    /// by a form.
    #[test]
    fn the_forms_name_pictures_that_exist() {
        fn named(fields: &[fieldcraft_engine::schema::Field], out: &mut Vec<&'static str>, depth: usize) {
            for f in fields {
                if !f.picture.is_empty() {
                    out.push(f.picture);
                }
                if let fieldcraft_engine::schema::Kind::Group(inner) = f.kind
                    && depth < fieldcraft_engine::schema::MAX_DEPTH
                {
                    named(inner, out, depth + 1);
                }
            }
        }
        let mut used = Vec::new();
        for c in fieldcraft_engine::commands() {
            named(c.form.unwrap_or_default(), &mut used, 0);
        }
        for name in &used {
            assert!(NAMES.contains(name), "{name}");
        }
        for name in NAMES {
            assert!(used.contains(name), "{name} is in no form");
        }
    }
}
