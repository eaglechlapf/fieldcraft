//! The unknowns of the grid: which shape function of which cell is which global mode.
//!
//! A cell of degree p carries the tensor product of the one-dimensional shape functions
//! ([`crate::legendre`]): (p + 1)³ modes, mode `(a, b, c)` at local index
//! `a + (p + 1)(b + (p + 1) c)`. Along one axis, index 0 belongs to the cell's lower grid plane,
//! index 1 to its upper one and an index of 2 or more to the cell's own span, so a mode with no
//! such index sits on a grid vertex, with one on an edge, with two on a face and with three in
//! the cell's interior. Two cells share the modes of the vertices, edges and faces they share:
//! the grid is Cartesian, so both see such a mode with the same orientation and no sign is
//! needed. A mode is named by its level, where it sits (in half cell steps of that level) and
//! its degree along each axis.
//!
//! Every mode carries the three displacement components, so to the sparse matrix it is a node
//! with three translations.
//!
//! # The trunk space
//! The full tensor product holds every monomial ξⁱ ηʲ ζᵏ with i, j, k ≤ p, far more than the
//! polynomials of total degree p that set the rate of convergence. The *trunk space* (B. Szabó,
//! I. Babuška, *Finite Element Analysis*, Wiley 1991, the hexahedral element) keeps the vertex
//! modes, the edge modes up to degree p, the face modes whose two degrees sum to at most p and
//! the interior modes whose three degrees sum to at most p: it still holds every polynomial of
//! total degree p, with a fraction of the modes (74 instead of 216 at p = 5). Two cells keep the
//! same modes of a shared face, so the sharing above is unchanged.
//!
//! # Refined cells
//! Where cells are divided ([`crate::refine`]) the cells of every level have such modes, and
//! the unknowns are the active ones of all levels. The cells the part is integrated over, the
//! *elements*, are the leaves; the functions that reach an element are its own active modes
//! and the active modes of the cells above it, given on the element's modes by its
//! [`Map`]. An element of the grid that no refinement touches has its own modes and no map.
//!
//! # Islands
//! A cell whose material is in separate pieces ([`crate::islands`]) is an element once per
//! island. Which functions an element has, and which of them are the same unknown as another
//! element's, is then decided by the material and not by the grid alone ([`Gaps`]): around a
//! vertex, an edge, a face or an interior of a level, the elements that material joins there
//! are a group, the rules of [`crate::refine`] are read for each group on its own, and a
//! function is one unknown per group. So the two sides of a gap share nothing where nothing
//! joins them, each side keeps a complete basis of its own (a side that is refined alone has
//! the functions of its own finest cells, not those of a coarser cell across the gap), and
//! where no gap is near, everything is as it is without islands, to the numbering.

use std::collections::{BTreeMap, BTreeSet};

use crate::legendre::MAX_DEGREE;
use crate::refine::{ALL_PARTS, Map, Transfer, Tree, part_of};
use crate::{Error, Result};

/// What a cell is to the part.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    Outside,
    Inside,
    Cut,
}

/// The slot of a cell that carries no unknowns.
pub(crate) const NONE: u32 = u32::MAX;

/// Cells whose active parts are found per parallel task.
const GRAIN: usize = 1024;

#[derive(Debug, Clone)]
pub(crate) struct Space {
    /// Modes per cell.
    pub modes: usize,
    /// The elements: the leaves of the tree that hold material, by node, in the tree's order,
    /// a leaf once for each of its islands.
    pub elements: Vec<u32>,
    /// Which island of its cell each element is.
    island: Vec<u16>,
    /// For each node of the tree, its first element (the one of island 0), or [`NONE`].
    pub element_of: Vec<u32>,
    /// Where the functions of each element start in `functions`, and the end of the last.
    start: Vec<usize>,
    /// The global mode of every function of every element.
    functions: Vec<usize>,
    /// For each element, its functions on the modes of its cell; `None` where they are those
    /// modes themselves, in their order.
    maps: Vec<Option<Map>>,
    /// Global modes.
    pub count: usize,
    /// The modes that are there because material is in separate pieces: one for every group of
    /// elements a function is an unknown of, after the first.
    pub told_apart: usize,
    /// Where any are: for each element, where the elements its material goes on into start in
    /// the list, and the list ([`Space::around`]).
    joined: Option<(Vec<u32>, Vec<u32>)>,
    /// Where each mode sits, in half cell steps of the grid from its origin.
    pub xyz: Vec<[f64; 3]>,
}

/// The modes of a cell of degree `p`, each by its index along the three axes: the full tensor
/// product in the order `a + (p + 1)(b + (p + 1) c)`, or its trunk space in the same order.
pub(crate) fn local_modes(p: usize, trunk: bool) -> Vec<[u8; 3]> {
    let q = p.min(MAX_DEGREE) + 1;
    // The linear functions do not count towards the degree of a mode.
    let degree = |a: usize| if a < 2 { 0 } else { a };
    let mut modes = Vec::with_capacity(q * q * q);
    for c in 0..q {
        for b in 0..q {
            for a in 0..q {
                if !trunk || degree(a) + degree(b) + degree(c) <= p {
                    modes.push([a as u8, b as u8, c as u8]);
                }
            }
        }
    }
    modes
}

/// Which elements material joins, and which cells are near a gap: what [`Space::with_islands`]
/// reads the rules of the functions from where they are not the grid's alone.
struct Gaps {
    /// Where the neighbours of each element start in `to`, and the end of the last: the
    /// elements its material goes on into ([`crate::islands::links`]).
    start: Vec<u32>,
    to: Vec<u32>,
    /// For each node of the tree, whether it or a cell under it is a leaf of several islands or
    /// a leaf that some other leaf across one of its faces is not joined to.
    near: Vec<bool>,
    /// The groups around the vertices, edges, faces and interiors that were asked for, by level
    /// and place ([`Gaps::groups`]).
    found: BTreeMap<[u32; 4], Vec<Member>>,
}

/// An element among those around a vertex, an edge, a face or an interior of a level.
#[derive(Debug, Clone, Copy)]
struct Member {
    element: u32,
    /// The lowest element of its group.
    group: u32,
    /// Whether the functions of that place are unknowns of the group: a leaf of the level is in
    /// it and no leaf of a lower level is.
    active: bool,
}

impl Gaps {
    fn new(tree: &Tree, space: &Space, links: &[[u32; 4]]) -> Gaps {
        let elements = space.elements.len();
        let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(2 * links.len());
        for link in links {
            let ends = (space.element(link[0] as usize, link[1] as usize), space.element(link[2] as usize, link[3] as usize));
            if let (Some(a), Some(b)) = ends
                && let (Ok(a), Ok(b)) = (u32::try_from(a), u32::try_from(b))
            {
                pairs.extend([(a, b), (b, a)]);
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        let mut start = vec![0u32; elements + 1];
        for (from, _) in &pairs {
            if let Some(count) = start.get_mut(*from as usize + 1) {
                *count += 1;
            }
        }
        let mut total = 0;
        for count in &mut start {
            total += *count;
            *count = total;
        }
        let mut gaps = Gaps { start, to: pairs.iter().map(|(_, to)| *to).collect(), near: vec![false; tree.nodes.len()], found: BTreeMap::new() };
        // The leaves at a gap: of several islands, or with an island that an island of a leaf
        // across a face is not joined to.
        let islands = |node: usize| (0..usize::from(u16::MAX)).map_while(|island| space.element(node, island)).collect::<Vec<usize>>();
        for (node, cell) in tree.nodes.iter().enumerate() {
            if !(cell.is_leaf() && cell.material()) {
                continue;
            }
            let own = islands(node);
            let mut at_gap = own.len() > 1;
            let level = usize::from(cell.level);
            for axis in 0..3 {
                for step in [-1i64, 1] {
                    let mut beside = cell.ijk.map(i64::from);
                    beside[axis] += step;
                    let Some(other) = tree.covering(level, beside) else { continue };
                    if !tree.nodes.get(other).is_some_and(|other| other.is_leaf() && other.material()) {
                        continue;
                    }
                    let others = islands(other);
                    if own.iter().any(|a| others.iter().any(|b| !gaps.joined(*a).contains(&(*b as u32)))) {
                        at_gap = true;
                        if let Some(flag) = gaps.near.get_mut(other) {
                            *flag = true;
                        }
                    }
                }
            }
            if at_gap && let Some(flag) = gaps.near.get_mut(node) {
                *flag = true;
            }
        }
        // And the cells above them: children come after their parents.
        for node in (0..tree.nodes.len()).rev() {
            if gaps.near.get(node).copied().unwrap_or(false)
                && let Some(parent) = tree.nodes.get(node).map(|cell| cell.parent as usize)
                && let Some(flag) = gaps.near.get_mut(parent)
            {
                *flag = true;
            }
        }
        gaps
    }

    /// The elements the material of `element` goes on into.
    fn joined(&self, element: usize) -> &[u32] {
        match (self.start.get(element), self.start.get(element + 1)) {
            (Some(&from), Some(&to)) => self.to.get(from as usize..to as usize).unwrap_or_default(),
            _ => &[],
        }
    }

    /// The cells of `level` around the part `part` of cell `ijk` of that level (a vertex, an
    /// edge, a face or the interior), as places among the cells of the level.
    fn around(ijk: [u32; 3], part: usize) -> impl Iterator<Item = [i64; 3]> {
        let along = |class: usize, i: i64| match class {
            0 => [Some(i - 1), Some(i)],
            1 => [Some(i), Some(i + 1)],
            _ => [Some(i), None],
        };
        let at = ijk.map(i64::from);
        let (xs, ys, zs) = (along(part % 3, at[0]), along((part / 3) % 3, at[1]), along(part / 9, at[2]));
        zs.into_iter().flatten().flat_map(move |z| ys.into_iter().flatten().flat_map(move |y| xs.into_iter().flatten().map(move |x| [x, y, z])))
    }

    /// The level and the place of the part `part` of cell `ijk` of `level`, in half cells.
    fn key(level: u8, ijk: [u32; 3], part: usize) -> [u32; 4] {
        let place = |class: usize, i: u32| match class {
            0 => 2 * i,
            1 => 2 * i + 2,
            _ => 2 * i + 1,
        };
        [u32::from(level), place(part % 3, ijk[0]), place((part / 3) % 3, ijk[1]), place(part / 9, ijk[2])]
    }

    /// Whether a gap is near the part `part` of the cell `node`: only then do its functions
    /// follow the groups.
    fn is_near(&self, tree: &Tree, node: usize, part: usize) -> bool {
        let Some(cell) = tree.nodes.get(node) else { return false };
        Gaps::around(cell.ijk, part)
            .any(|at| tree.covering(usize::from(cell.level), at).is_some_and(|other| self.near.get(other).copied().unwrap_or(false)))
    }

    /// The group of `element` around the part `part` of the cell `node`, which is the element's
    /// own cell or one above it: the lowest element of the group, and whether the functions of
    /// the part are unknowns of the group.
    ///
    /// The elements around the part are those of the leaves that the cells of the level around
    /// it are, lie under or lie in; two of them are in one group when material joins them, from
    /// one to the next, without leaving those leaves. The rules of [`crate::refine`] are then
    /// the group's own: the functions are active when a leaf of the level is in the group and
    /// no leaf of a lower level is.
    fn group(&mut self, tree: &Tree, space: &Space, node: usize, part: usize, element: usize) -> Option<(u32, bool)> {
        let cell = tree.nodes.get(node)?;
        let level = usize::from(cell.level);
        let key = Gaps::key(cell.level, cell.ijk, part);
        if !self.found.contains_key(&key) {
            // The elements around the part, each with the level of its leaf against the part's:
            // lower (0), the same (1) or higher (2).
            let mut around: Vec<(u32, u8)> = Vec::new();
            let mut under = Vec::new();
            for at in Gaps::around(cell.ijk, part) {
                let Some(top) = tree.covering(level, at) else { continue };
                under.push(top);
                while let Some(next) = under.pop() {
                    let Some(other) = tree.nodes.get(next).filter(|other| other.material()) else { continue };
                    if !other.is_leaf() {
                        under.extend((0..8).map(|child| other.children as usize + child));
                        continue;
                    }
                    let kind = match usize::from(other.level).cmp(&level) {
                        std::cmp::Ordering::Less => 0,
                        std::cmp::Ordering::Equal => 1,
                        std::cmp::Ordering::Greater => 2,
                    };
                    around.extend((0..usize::from(u16::MAX)).map_while(|island| space.element(next, island)).map(|element| (element as u32, kind)));
                }
            }
            around.sort_unstable();
            around.dedup();
            let index = |element: u32| around.binary_search_by(|(other, _)| other.cmp(&element)).ok();
            let mut members: Vec<Member> = around.iter().map(|(element, _)| Member { element: *element, group: NONE, active: false }).collect();
            for first in 0..around.len() {
                if members.get(first).is_none_or(|member| member.group != NONE) {
                    continue;
                }
                // The elements material joins to this one, among those around the part.
                let lowest = around.get(first).map_or(NONE, |(element, _)| *element);
                let mut group = vec![first];
                let mut front = vec![first];
                if let Some(member) = members.get_mut(first) {
                    member.group = lowest;
                }
                while let Some(at) = front.pop() {
                    let Some(&(from, _)) = around.get(at) else { continue };
                    for to in self.joined(from as usize) {
                        if let Some(other) = index(*to)
                            && let Some(member) = members.get_mut(other).filter(|member| member.group == NONE)
                        {
                            member.group = lowest;
                            group.push(other);
                            front.push(other);
                        }
                    }
                }
                let kinds = |kind: u8| group.iter().any(|member| around.get(*member).is_some_and(|(_, found)| *found == kind));
                let active = kinds(1) && !kinds(0);
                for member in group {
                    if let Some(member) = members.get_mut(member) {
                        member.active = active;
                    }
                }
            }
            self.found.insert(key, members);
        }
        let members = self.found.get(&key)?;
        let member = members.binary_search_by(|member| (member.element as usize).cmp(&element)).ok().and_then(|at| members.get(at))?;
        Some((member.group, member.active))
    }
}

/// The leaves whose functions hold the two sides of a gap together, where they are more than
/// `gap` widths of the gap across: the cells to divide for [`crate::Options::gaps`], in
/// increasing order.
///
/// Where a gap ends, material joins its two sides around the root, and a function whose cells
/// reach from the root along the gap is an unknown of both sides at once ([`Gaps::group`]): it
/// holds them together for as far as its cells reach. Those cells are the leaves of the
/// function's own level in its group; dividing them hands the function down to smaller cells,
/// which reach less far. `islands` gives the islands of a leaf by its node, `links` the islands
/// that material joins and `sides` the two sides of every gap ([`crate::islands`]); `size` is
/// the size of a cell of a level.
pub(crate) fn holding(
    tree: &Tree,
    islands: &dyn Fn(u32) -> usize,
    links: &[[u32; 4]],
    sides: &[crate::islands::Sides],
    gap: f64,
    size: &dyn Fn(usize) -> f64,
) -> Vec<usize> {
    let space = Space::elements(tree, islands);
    let mut gaps = Gaps::new(tree, &space, links);
    let mut divided: BTreeSet<usize> = BTreeSet::new();
    for side in sides {
        let [node, island, other_node, other_island] = side.islands;
        let (Some(first), Some(second)) = (space.element(node as usize, island as usize), space.element(other_node as usize, other_island as usize))
        else {
            continue;
        };
        // The cells whose functions reach the first side: its own and those above it.
        let mut at = node;
        while let Some(cell) = tree.nodes.get(at as usize) {
            if gap * side.width < size(usize::from(cell.level)) {
                for part in 0..27 {
                    let (Some((group, true)), Some((other, true))) =
                        (gaps.group(tree, &space, at as usize, part, first), gaps.group(tree, &space, at as usize, part, second))
                    else {
                        continue;
                    };
                    if group != other {
                        continue;
                    }
                    // The leaves of the function's level in the group carry it.
                    for member in gaps.found.get(&Gaps::key(cell.level, cell.ijk, part)).map(Vec::as_slice).unwrap_or_default() {
                        if member.group == group
                            && let Some(&leaf) = space.elements.get(member.element as usize)
                            && tree
                                .nodes
                                .get(leaf as usize)
                                .is_some_and(|leaf| leaf.level == cell.level && usize::from(leaf.level) < crate::grid::MAX_REFINE)
                        {
                            divided.insert(leaf as usize);
                        }
                    }
                }
            }
            at = cell.parent;
        }
    }
    divided.into_iter().collect()
}

impl Space {
    /// The elements of `tree` with no functions yet: the leaves that hold material (a cell that
    /// holds none is no element: a function of a cell above it that reaches it is not
    /// integrated there), once for each of its `islands`.
    fn elements(tree: &Tree, islands: &dyn Fn(u32) -> usize) -> Space {
        let mut space = Space {
            modes: 0,
            elements: Vec::new(),
            island: Vec::new(),
            element_of: vec![NONE; tree.nodes.len()],
            start: vec![0],
            functions: Vec::new(),
            maps: Vec::new(),
            count: 0,
            told_apart: 0,
            joined: None,
            xyz: Vec::new(),
        };
        for leaf in tree.leaves() {
            if !tree.nodes.get(leaf as usize).is_some_and(|cell| cell.material()) {
                continue;
            }
            if let Some(element) = space.element_of.get_mut(leaf as usize) {
                *element = space.elements.len() as u32;
            }
            for island in 0..islands(leaf).clamp(1, usize::from(u16::MAX)) {
                space.elements.push(leaf);
                space.island.push(island as u16);
            }
        }
        space
    }

    /// The unknowns of the cells of `tree`, each cell carrying the modes `local`
    /// ([`local_modes`]) of degree `p` and being one element.
    #[cfg(test)]
    pub fn new(tree: &Tree, local: &[[u8; 3]], p: usize) -> Space {
        Space::with_islands(tree, local, p, &|_| 1, None)
    }

    /// [`Space::new`] with every leaf an element once for each of its `islands` (by node; at
    /// least one). With `links` (every two elements whose material goes on from the one into
    /// the other through the face their cells share, `[node, island, node, island]`:
    /// [`crate::islands::links`]) the functions follow the material near a gap (the module's
    /// *Islands*); without, all elements of a cell have its functions and two cells share the
    /// functions of what they share, whatever they hold.
    pub fn with_islands(tree: &Tree, local: &[[u8; 3]], p: usize, islands: &dyn Fn(u32) -> usize, links: Option<&[[u32; 4]]>) -> Space {
        let mut space = Space::elements(tree, islands);
        space.modes = local.len();
        let mut gaps = links.map(|links| Gaps::new(tree, &space, links)).filter(|gaps| gaps.near.iter().any(|near| *near));
        let transfer = Transfer::new(p, tree.deepest());
        let parts: Vec<u32> = if tree.deepest() == 0 {
            vec![ALL_PARTS; tree.nodes.len()]
        } else {
            fieldcraft_parallel::map_grained(tree.nodes.len(), GRAIN, |node| tree.active_parts(node))
        };
        let parts = |node: u32| parts.get(node as usize).copied().unwrap_or(0);
        let active = |parts: u32, mode: &[u8; 3]| (parts >> part_of(mode)) & 1 == 1;
        // Where along one axis local index `a` of cell `i` sits, and its degree there (0 for the
        // two linear functions, which differ by where they sit).
        let place = |i: u32, a: usize| -> (u32, u32) {
            match a {
                0 => (2 * i, 0),
                1 => (2 * i + 2, 0),
                _ => (2 * i + 1, a as u32),
            }
        };
        // A function by its level, where it sits, its degrees and, near a gap, its group.
        let mut names: BTreeMap<[u32; 8], usize> = BTreeMap::new();
        let mut name = |space: &mut Space, level: u8, ijk: [u32; 3], mode: &[u8; 3], group: u32| {
            let [a, b, c] = mode.map(usize::from);
            let ((x, dx), (y, dy), (z, dz)) = (place(ijk[0], a), place(ijk[1], b), place(ijk[2], c));
            let next = names.len();
            let found = *names.entry([u32::from(level), x, y, z, dx, dy, dz, group]).or_insert(next);
            if found == next {
                let steps = (1u64 << level) as f64;
                space.xyz.push([x as f64 / steps, y as f64 / steps, z as f64 / steps]);
            }
            space.functions.push(found);
        };
        for element in 0..space.elements.len() {
            let Some(&leaf) = space.elements.get(element) else { continue };
            let Some(cell) = tree.nodes.get(leaf as usize) else { continue };
            // Which parts of a cell (the element's own or one above it) carry functions of the
            // element, and the group of each: the grid's rule, or near a gap the group's.
            let mut reached = |space: &Space, node: u32| -> (u32, [u32; 27]) {
                let (mut found, mut groups) = (parts(node), [NONE; 27]);
                if let Some(gaps) = gaps.as_mut() {
                    for part in 0..27 {
                        if !gaps.is_near(tree, node as usize, part) {
                            continue;
                        }
                        found &= !(1 << part);
                        if let Some((group, true)) = gaps.group(tree, space, node as usize, part, element) {
                            found |= 1 << part;
                            groups[part] = group;
                        }
                    }
                }
                (found, groups)
            };
            let (own, own_groups) = reached(&space, leaf);
            // The cells above that have active modes.
            let mut above = Vec::new();
            let mut parent = cell.parent;
            while let Some(ancestor) = tree.nodes.get(parent as usize) {
                let (found, groups) = reached(&space, parent);
                if found != 0 {
                    above.push((found, groups, *ancestor));
                }
                parent = ancestor.parent;
            }
            if own == ALL_PARTS && above.is_empty() && own_groups.iter().all(|group| *group == NONE) {
                for mode in local {
                    name(&mut space, cell.level, cell.ijk, mode, NONE);
                }
                space.maps.push(None);
            } else {
                let mut map = Map::default();
                for (index, mode) in local.iter().enumerate() {
                    if active(own, mode) {
                        name(&mut space, cell.level, cell.ijk, mode, own_groups[part_of(mode)]);
                        map.push([(index, 1.0)].into_iter());
                    }
                }
                for (reached, groups, ancestor) in above {
                    let down = usize::from(cell.level.saturating_sub(ancestor.level));
                    let part = cell.ijk.map(|i| (i & ((1u32 << down) - 1)) as usize);
                    for mode in local {
                        if !active(reached, mode) {
                            continue;
                        }
                        let rows = [0, 1, 2].map(|axis| transfer.row(down, part[axis], usize::from(mode[axis])));
                        name(&mut space, ancestor.level, ancestor.ijk, mode, groups[part_of(mode)]);
                        map.push(local.iter().enumerate().filter_map(|(index, fine)| {
                            let weight = (0..3).map(|axis| rows[axis].get(usize::from(fine[axis])).copied().unwrap_or(0.0)).product::<f64>();
                            (weight != 0.0).then_some((index, weight))
                        }));
                    }
                }
                space.maps.push(Some(map));
            }
            space.start.push(space.functions.len());
        }
        space.count = names.len();
        // The functions that are an unknown once per group: every one of a place and degree
        // after the first.
        let mut before: Option<[u32; 7]> = None;
        for key in names.keys() {
            let place = [key[0], key[1], key[2], key[3], key[4], key[5], key[6]];
            if before == Some(place) {
                space.told_apart += 1;
            }
            before = Some(place);
        }
        // Where nothing was told apart, nothing asks what joins what.
        if space.told_apart > 0
            && let Some(gaps) = gaps
        {
            space.joined = Some((gaps.start, gaps.to));
        }
        space
    }

    /// Where functions were told apart: `element` and the elements up to `hops` links of
    /// material from it, in increasing order. `None` where none were: every cell around a cell
    /// is then as near as the grid says.
    pub fn around(&self, element: usize, hops: usize) -> Option<Vec<u32>> {
        let (start, to) = self.joined.as_ref()?;
        let joined = |element: u32| match (start.get(element as usize), start.get(element as usize + 1)) {
            (Some(&from), Some(&end)) => to.get(from as usize..end as usize).unwrap_or_default(),
            _ => &[],
        };
        let element = u32::try_from(element).ok()?;
        let mut found = vec![element];
        let mut front = vec![element];
        for _ in 0..hops {
            let mut next = Vec::new();
            for from in &front {
                for to in joined(*from) {
                    if !found.contains(to) {
                        found.push(*to);
                        next.push(*to);
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            front = next;
        }
        found.sort_unstable();
        Some(found)
    }

    /// The element of island `island` of the cell `node`.
    pub fn element(&self, node: usize, island: usize) -> Option<usize> {
        let first = self.element_of.get(node).copied().filter(|element| *element != NONE)? as usize;
        let element = first.checked_add(island)?;
        (self.elements.get(element).copied() == u32::try_from(node).ok()).then_some(element)
    }

    /// Which island of its cell `element` is.
    pub fn island(&self, element: usize) -> usize {
        self.island.get(element).copied().map_or(0, usize::from)
    }

    /// The global modes of the functions of `element`.
    pub fn functions(&self, element: usize) -> &[usize] {
        match (self.start.get(element), self.start.get(element + 1)) {
            (Some(&from), Some(&to)) => self.functions.get(from..to).unwrap_or_default(),
            _ => &[],
        }
    }

    /// The functions of `element` on the modes of its cell; `None` where they are the modes.
    pub fn map(&self, element: usize) -> Option<&Map> {
        self.maps.get(element).and_then(Option::as_ref)
    }

    /// The coefficients of the modes of the cell of `element`, three per mode, in the solution
    /// `u` (three per global mode).
    pub fn local(&self, element: usize, u: &[f64]) -> Vec<f64> {
        let own: Vec<f64> =
            self.functions(element).iter().flat_map(|mode| (0..3).map(move |i| u.get(3 * mode + i).copied().unwrap_or(0.0))).collect();
        match self.map(element) {
            Some(map) => map.local(self.modes, &own),
            None => own,
        }
    }

    /// Renumber the modes so that `order[k]` becomes mode `k`.
    pub fn renumber(&mut self, order: &[usize]) -> Result<()> {
        let broken = || Error::Internal("the mode order is not a permutation".into());
        if order.len() != self.count {
            return Err(broken());
        }
        let mut position = vec![usize::MAX; self.count];
        let mut xyz = Vec::with_capacity(self.count);
        for (k, &mode) in order.iter().enumerate() {
            let slot = position.get_mut(mode).ok_or_else(broken)?;
            if *slot != usize::MAX {
                return Err(broken());
            }
            *slot = k;
            xyz.push(self.xyz.get(mode).copied().ok_or_else(broken)?);
        }
        for mode in &mut self.functions {
            *mode = position.get(*mode).copied().ok_or_else(broken)?;
        }
        self.xyz = xyz;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;

    /// Two cells share the functions of the face between them where material goes on through
    /// it, and have their own where it does not: the two sides of a gap that lies in the face.
    #[test]
    fn cells_that_no_material_joins_share_nothing() {
        let grid = Grid { origin: [0.0; 3], spacing: 1.0, cells: [2, 1, 1] };
        let tree = Tree::new(&grid, &[Kind::Cut, Kind::Cut]);
        for p in 1..=3 {
            let (q, local) = (p + 1, local_modes(p, false));
            let joined = Space::with_islands(&tree, &local, p, &|_| 1, Some(&[[0, 0, 1, 0]]));
            assert_eq!((joined.count, joined.told_apart), (2 * q * q * q - q * q, 0), "degree {p}");
            assert!(joined.around(0, 3).is_none());
            let apart = Space::with_islands(&tree, &local, p, &|_| 1, Some(&[]));
            assert_eq!((apart.count, apart.told_apart), (2 * q * q * q, q * q), "degree {p}");
            assert!(apart.functions(0).iter().all(|mode| !apart.functions(1).contains(mode)));
            assert_eq!(apart.around(0, 3), Some(vec![0]));
            // Two islands in the first cell, of which the second goes on into the second cell.
            let islands = Space::with_islands(&tree, &local, p, &|node| if node == 0 { 2 } else { 1 }, Some(&[[0, 1, 1, 0]]));
            assert_eq!(islands.elements, vec![0, 0, 1]);
            assert_eq!((islands.count, islands.told_apart), (3 * q * q * q - q * q, q * q * q), "degree {p}");
            assert!(islands.functions(0).iter().all(|mode| !islands.functions(1).contains(mode) && !islands.functions(2).contains(mode)));
            assert_eq!(islands.functions(1).iter().filter(|mode| islands.functions(2).contains(mode)).count(), q * q);
            assert_eq!((islands.around(1, 1), islands.around(0, 1)), (Some(vec![1, 2]), Some(vec![0])));
        }
    }

    #[test]
    fn neighbours_share_the_modes_of_their_common_face() {
        let grid = Grid { origin: [0.0; 3], spacing: 1.0, cells: [2, 1, 1] };
        for p in 1..=4 {
            let q = p + 1;
            let mut space = Space::new(&Tree::new(&grid, &[Kind::Inside, Kind::Cut]), &local_modes(p, false), p);
            // Two cells of q³ modes each, sharing the q² modes of one face.
            assert_eq!(space.count, 2 * q * q * q - q * q, "degree {p}");
            assert_eq!(space.xyz.len(), space.count);
            assert!(space.map(0).is_none() && space.map(1).is_none());
            let (left, right) = (space.functions(0).to_vec(), space.functions(1).to_vec());
            for c in 0..q {
                for b in 0..q {
                    // The upper x plane of the left cell is the lower x plane of the right one.
                    assert_eq!(left[1 + q * (b + q * c)], right[q * (b + q * c)]);
                }
            }
            // Reversing the numbering keeps the sharing.
            let order: Vec<usize> = (0..space.count).rev().collect();
            space.renumber(&order).unwrap();
            assert_eq!(space.functions(0)[1], space.functions(1)[0]);
            assert_eq!(space.functions(0)[0], space.count - 1 - left[0]);
            assert!(space.renumber(&[0, 0]).is_err());
        }
        let tree = Tree::new(&grid, &[Kind::Outside, Kind::Inside]);
        let space = Space::new(&tree, &local_modes(2, false), 2);
        assert_eq!((space.elements.clone(), tree.root.clone(), space.count), (vec![0], vec![NONE, 0], 27));
    }

    /// The trunk space has the modes Szabó and Babuška count: 8 vertex modes, p − 1 an edge,
    /// (p − 2)(p − 3)/2 a face and (p − 3)(p − 4)(p − 5)/6 inside, and two cells share the ones
    /// of their common face.
    #[test]
    fn the_trunk_space_keeps_the_modes_of_total_degree() {
        for p in 1..=MAX_DEGREE {
            let trunk = local_modes(p, true);
            let (face, inside) = (if p >= 4 { (p - 2) * (p - 3) / 2 } else { 0 }, if p >= 6 { (p - 3) * (p - 4) * (p - 5) / 6 } else { 0 });
            assert_eq!(trunk.len(), 8 + 12 * (p - 1) + 6 * face + inside, "degree {p}");
            assert_eq!(local_modes(p, false).len(), (p + 1) * (p + 1) * (p + 1));
            // Every vertex mode, and in the order of the tensor product.
            assert!((0..8u8).all(|corner| trunk.contains(&[corner & 1, (corner >> 1) & 1, (corner >> 2) & 1])));
            assert!(trunk.windows(2).all(|w| (w[0][2], w[0][1], w[0][0]) < (w[1][2], w[1][1], w[1][0])));
            let grid = Grid { origin: [0.0; 3], spacing: 1.0, cells: [2, 1, 1] };
            let space = Space::new(&Tree::new(&grid, &[Kind::Inside, Kind::Cut]), &trunk, p);
            let on_face = trunk.iter().filter(|mode| mode[0] == 1).count();
            assert_eq!(space.count, 2 * trunk.len() - on_face, "degree {p}");
        }
    }
}
