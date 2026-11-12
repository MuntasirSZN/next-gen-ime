//! Interactive IME-style demo for `upodesh`.
//!
//! Type Roman text (Avro or Khipro); `okkhor` converts it to the Bangla
//! preedit and `upodesh` ranks suggestions for it. Every `CorrectionConfig`
//! and `EditCosts` knob is live, and each keystroke reports latency and
//! allocator usage.
//!
//! Usage:
//!   cargo run --release --manifest-path gui/Cargo.toml
//!
//! Bangla needs a font with Bengali glyphs. The demo looks for
//! Kohinoor Bangla (macOS) or Noto Sans Bengali (Linux); override with
//! `UPODESH_FONT=/path/to/font.ttf UPODESH_FONT_FAMILY="Family Name"`.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use iced::widget::{
    Column, button, checkbox, column, container, horizontal_rule, radio, row, scrollable, slider,
    text, text_input,
};
use iced::{Element, Font, Length, Task, Theme};
use okkhor::khipro::KhiproPhonetic;
use okkhor::parser::Parser;
use peak_alloc::PeakAlloc;
use upodesh::{
    ClassWeights, CorrectionConfig, Corrector, EditCosts, IndexStats, InputMode, Suggestion,
    SuggestionInput,
};

#[global_allocator]
static PEAK_ALLOC: PeakAlloc = PeakAlloc;

const WEIGHTS_JSON: &[u8] = include_bytes!("../../data/weights.json");
const WINDOW: usize = 256;
const BENCH_RUNS: usize = 1000;
const INPUT_ID: &str = "roman-input";

fn main() -> iced::Result {
    let font = load_bangla_font();
    let mut app = iced::application("upodesh demo", App::update, App::view)
        .theme(|_| Theme::Dark)
        .window_size((1360.0, 860.0));
    let family = font.as_ref().map(|(_, family)| *family);
    if let Some((bytes, family)) = font {
        app = app.font(bytes).default_font(Font::with_name(family));
    }
    app.run_with(move || {
        (
            App::new(family),
            text_input::focus(text_input::Id::new(INPUT_ID)),
        )
    })
}

/// Find a font with Bengali glyphs: iced's bundled font has none.
fn load_bangla_font() -> Option<(Vec<u8>, &'static str)> {
    if let Ok(path) = std::env::var("UPODESH_FONT") {
        let family = std::env::var("UPODESH_FONT_FAMILY")
            .unwrap_or_else(|_| "Noto Sans Bengali".to_string());
        if let Ok(bytes) = std::fs::read(&path) {
            return Some((bytes, Box::leak(family.into_boxed_str())));
        }
    }
    const CANDIDATES: &[(&str, &str)] = &[
        ("/System/Library/Fonts/KohinoorBangla.ttc", "Kohinoor Bangla"),
        (
            "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf",
            "Noto Sans Bengali",
        ),
        (
            "/usr/share/fonts/noto/NotoSansBengali-Regular.ttf",
            "Noto Sans Bengali",
        ),
        (
            "/usr/share/fonts/google-noto/NotoSansBengali-Regular.ttf",
            "Noto Sans Bengali",
        ),
    ];
    CANDIDATES
        .iter()
        .find_map(|(path, family)| std::fs::read(path).ok().map(|bytes| (bytes, *family)))
}

#[derive(Debug, Clone)]
enum Message {
    RomanChanged(String),
    ModeChanged(InputMode),
    Knob(Knob),
    ResetDefaults,
    Benchmark,
    Pick(usize),
    ClearCommitted,
}

#[derive(Debug, Clone, Copy)]
enum Knob {
    MaxEditCost(u16),
    Limit(usize),
    FrequencyWeight(f64),
    LanguageModelWeight(f64),
    RomanWeight(f64),
    MaxRomanCost(u16),
    Insertion(u16),
    Deletion(u16),
    Substitution(u16),
    Transposition(u16),
    ClassWeights(bool),
}

/// User-facing knob values; the corrector is rebuilt from these.
#[derive(Clone)]
struct Knobs {
    config: CorrectionConfig,
    insertion: u16,
    deletion: u16,
    substitution: u16,
    transposition: u16,
    class_weights: bool,
}

impl Default for Knobs {
    fn default() -> Self {
        Self {
            config: CorrectionConfig::default(),
            insertion: 100,
            deletion: 100,
            substitution: 100,
            transposition: 65,
            class_weights: false,
        }
    }
}

impl Knobs {
    fn edit_costs(&self) -> EditCosts {
        let costs = EditCosts::bengali().with_operation_costs(
            self.insertion,
            self.deletion,
            self.substitution,
            self.transposition,
        );
        if self.class_weights {
            costs.with_class_weights(&ClassWeights::from_json(WEIGHTS_JSON))
        } else {
            costs
        }
    }
}

/// Rolling latency samples.
#[derive(Default)]
struct Samples {
    window: VecDeque<Duration>,
    last: Duration,
}

impl Samples {
    fn push(&mut self, sample: Duration) {
        if self.window.len() == WINDOW {
            self.window.pop_front();
        }
        self.window.push_back(sample);
        self.last = sample;
    }
}

/// (p50, p95, p99, max) of a sample set.
fn percentiles(samples: impl IntoIterator<Item = Duration>) -> Option<[Duration; 4]> {
    let mut sorted: Vec<Duration> = samples.into_iter().collect();
    if sorted.is_empty() {
        return None;
    }
    sorted.sort();
    let at = |p: f64| sorted[((sorted.len() - 1) as f64 * p) as usize];
    Some([at(0.50), at(0.95), at(0.99), sorted[sorted.len() - 1]])
}

struct Bench {
    query: String,
    suggest: [Duration; 4],
    prefix: [Duration; 4],
}

struct App {
    /// Owned so `with_config`/`with_edit_costs` can be applied; `None` only
    /// transiently while a builder runs.
    corrector: Option<Corrector>,
    avro: Parser,
    khipro: KhiproPhonetic,
    mode: InputMode,
    /// Full Roman line in the textbox; only its last word is in preedit.
    line: String,
    /// Finished words: (Roman word, chosen Bangla), by position in `line`.
    committed: Vec<(String, String)>,
    /// The word being typed and its Bangla conversion.
    roman: String,
    bangla: String,
    suggestions: Vec<Suggestion>,
    completions: Vec<String>,
    knobs: Knobs,
    font_family: Option<&'static str>,
    load_error: Option<String>,

    stats: IndexStats,
    index_bytes: usize,
    load_time: Duration,
    convert: Samples,
    suggest: Samples,
    prefix: Samples,
    query_peak: usize,
    query_live: isize,
    bench: Option<Bench>,
}

impl App {
    fn new(font_family: Option<&'static str>) -> Self {
        let before = PEAK_ALLOC.current_usage();
        let started = Instant::now();
        let loaded = upodesh::load_embedded();
        let load_time = started.elapsed();
        let index_bytes = PEAK_ALLOC.current_usage().saturating_sub(before);
        let (corrector, load_error) = match loaded {
            Ok(corrector) => (corrector, None),
            Err(err) => (
                Corrector::from_words(std::iter::empty::<&str>()).expect("empty index builds"),
                Some(err.to_string()),
            ),
        };
        Self {
            stats: corrector.stats(),
            corrector: Some(corrector),
            avro: Parser::new_phonetic(),
            khipro: KhiproPhonetic::new(),
            mode: InputMode::Avro,
            line: String::new(),
            committed: Vec::new(),
            roman: String::new(),
            bangla: String::new(),
            suggestions: Vec::new(),
            completions: Vec::new(),
            knobs: Knobs::default(),
            font_family,
            load_error,
            index_bytes,
            load_time,
            convert: Samples::default(),
            suggest: Samples::default(),
            prefix: Samples::default(),
            query_peak: 0,
            query_live: 0,
            bench: None,
        }
    }

    fn corrector(&self) -> &Corrector {
        self.corrector.as_ref().expect("corrector is always restored")
    }

    fn rebuild(&mut self, f: impl FnOnce(Corrector) -> Corrector) {
        let corrector = self.corrector.take().expect("corrector is always restored");
        self.corrector = Some(f(corrector));
    }

    fn query_input(&self) -> SuggestionInput<'_> {
        match self.mode {
            InputMode::Bangla => SuggestionInput::bangla_only(&self.bangla),
            mode => SuggestionInput::with_roman(&self.roman, &self.bangla, mode),
        }
    }

    fn convert_roman(&mut self) {
        let started = Instant::now();
        self.bangla.clear();
        match self.mode {
            InputMode::Khipro => self.khipro.convert_into(&self.roman, &mut self.bangla),
            // Bangla-only mode still uses Avro for the preedit, but sends
            // `roman: None` so only the Bangla side is scored.
            InputMode::Avro | InputMode::Bangla => {
                self.avro.convert_into(&self.roman, &mut self.bangla)
            }
        }
        self.convert.push(started.elapsed());
    }

    fn run_query(&mut self) {
        let limit = self.knobs.config.max_results;
        // Drop old results first so they don't count against this query.
        self.suggestions = Vec::new();
        self.completions = Vec::new();

        let before = PEAK_ALLOC.current_usage();
        PEAK_ALLOC.reset_peak_usage();

        let started = Instant::now();
        let suggestions = self.corrector().suggest_input(&self.query_input(), limit);
        let suggest_time = started.elapsed();

        let started = Instant::now();
        let completions = self.corrector().complete_prefix(&self.bangla, limit);
        let prefix_time = started.elapsed();

        self.query_peak = PEAK_ALLOC.peak_usage().saturating_sub(before);
        self.query_live = PEAK_ALLOC.current_usage() as isize - before as isize;

        self.suggestions = suggestions;
        self.completions = completions;
        if !self.bangla.is_empty() {
            self.suggest.push(suggest_time);
            self.prefix.push(prefix_time);
        }
    }

    fn refresh(&mut self) {
        self.convert_roman();
        self.run_query();
    }

    fn benchmark(&mut self) {
        if self.bangla.is_empty() {
            return;
        }
        let limit = self.knobs.config.max_results;
        let corrector = self.corrector();
        let input = self.query_input();
        let time = |f: &dyn Fn()| -> Vec<Duration> {
            (0..BENCH_RUNS)
                .map(|_| {
                    let started = Instant::now();
                    f();
                    started.elapsed()
                })
                .collect()
        };
        let suggest = percentiles(time(&|| {
            std::hint::black_box(corrector.suggest_input(&input, limit));
        }));
        let prefix = percentiles(time(&|| {
            std::hint::black_box(corrector.complete_prefix(&self.bangla, limit));
        }));
        if let (Some(suggest), Some(prefix)) = (suggest, prefix) {
            self.bench = Some(Bench {
                query: format!("{} → {}", self.roman, self.bangla),
                suggest,
                prefix,
            });
        }
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::RomanChanged(roman) => self.input_changed(roman),
            Message::ModeChanged(mode) => {
                self.mode = mode;
                self.refresh();
                return text_input::focus(text_input::Id::new(INPUT_ID));
            }
            Message::Knob(knob) => {
                self.apply_knob(knob);
                self.run_query();
            }
            Message::ResetDefaults => {
                self.knobs = Knobs::default();
                let config = self.knobs.config.clone();
                let costs = self.knobs.edit_costs();
                self.rebuild(|c| c.with_config(config).with_edit_costs(costs));
                self.run_query();
            }
            Message::Benchmark => self.benchmark(),
            Message::Pick(index) => {
                if let Some(word) = self.chosen(index) {
                    self.committed.push((self.roman.clone(), word));
                    self.line.push(' ');
                    self.roman.clear();
                    self.refresh();
                }
                return text_input::focus(text_input::Id::new(INPUT_ID));
            }
            Message::ClearCommitted => {
                self.line.clear();
                self.committed.clear();
                self.roman.clear();
                self.refresh();
            }
        }
        Task::none()
    }

    /// Suggestion `index` for the current word, or the raw preedit when
    /// there is none.
    fn chosen(&self, index: usize) -> Option<String> {
        self.suggestions
            .get(index)
            .map(|s| s.word.clone())
            .or_else(|| (!self.bangla.is_empty()).then(|| self.bangla.clone()))
    }

    /// IME-style editing: a word followed by whitespace is finished and
    /// commits its top suggestion; the last word stays in preedit.
    ///
    /// iced may deliver several keystrokes built from a stale buffer, so this
    /// is a pure function of the line: choices are cached by word position
    /// and only recomputed when that position's Roman word changed.
    fn input_changed(&mut self, line: String) {
        let mut words: Vec<&str> = line.split_whitespace().collect();
        let current = if line.ends_with(char::is_whitespace) {
            ""
        } else {
            words.pop().unwrap_or("")
        };
        let unchanged = self
            .committed
            .iter()
            .zip(&words)
            .take_while(|((roman, _), word)| roman == *word)
            .count();
        self.committed.truncate(unchanged);
        for word in &words[unchanged..] {
            self.roman = word.to_string();
            self.refresh();
            let chosen = self.chosen(0).unwrap_or_default();
            self.committed.push((self.roman.clone(), chosen));
        }
        self.roman = current.to_string();
        self.line = line;
        self.refresh();
    }

    fn apply_knob(&mut self, knob: Knob) {
        let config = &mut self.knobs.config;
        let config_changed = match knob {
            Knob::MaxEditCost(v) => {
                config.max_edit_cost = v;
                true
            }
            Knob::Limit(v) => {
                // Read at query time; the corrector needn't change.
                config.max_results = v;
                false
            }
            Knob::FrequencyWeight(v) => {
                config.frequency_weight = v;
                true
            }
            Knob::LanguageModelWeight(v) => {
                config.language_model_weight = v;
                true
            }
            Knob::RomanWeight(v) => {
                config.roman_weight = v;
                true
            }
            Knob::MaxRomanCost(v) => {
                config.max_roman_cost = v;
                true
            }
            Knob::Insertion(v) => {
                self.knobs.insertion = v;
                false
            }
            Knob::Deletion(v) => {
                self.knobs.deletion = v;
                false
            }
            Knob::Substitution(v) => {
                self.knobs.substitution = v;
                false
            }
            Knob::Transposition(v) => {
                self.knobs.transposition = v;
                false
            }
            Knob::ClassWeights(v) => {
                self.knobs.class_weights = v;
                false
            }
        };
        if config_changed {
            let config = self.knobs.config.clone();
            self.rebuild(|c| c.with_config(config));
        } else if !matches!(knob, Knob::Limit(_)) {
            let costs = self.knobs.edit_costs();
            self.rebuild(|c| c.with_edit_costs(costs));
        }
    }

    fn view(&self) -> Element<'_, Message> {
        row![
            container(self.view_input()).width(Length::FillPortion(5)),
            container(scrollable(self.view_config())).width(Length::FillPortion(3)),
            container(scrollable(self.view_perf())).width(Length::FillPortion(3)),
        ]
        .spacing(20)
        .padding(16)
        .into()
    }

    fn view_input(&self) -> Element<'_, Message> {
        let modes = [
            ("Avro", InputMode::Avro),
            ("Khipro", InputMode::Khipro),
            ("Bangla only", InputMode::Bangla),
        ]
        .into_iter()
        .fold(row![text("Mode:")].spacing(14), |r, (label, mode)| {
            r.push(radio(label, mode, Some(self.mode), Message::ModeChanged))
        });

        let input = text_input("Type Roman text, e.g. amar sonar bangla (space commits)", &self.line)
            .id(text_input::Id::new(INPUT_ID))
            .on_input(Message::RomanChanged)
            .on_submit(Message::Pick(0))
            .size(20)
            .padding(10);

        let preedit = bn(if self.bangla.is_empty() {
            "—".to_string()
        } else {
            self.bangla.clone()
        })
        .size(34);

        let committed = row![
            bn(if self.committed.is_empty() {
                "Committed text appears here (Space/Enter or click a suggestion)".to_string()
            } else {
                self.committed
                    .iter()
                    .map(|(_, bangla)| bangla.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .size(20)
            .width(Length::Fill),
            button("Clear")
                .style(button::secondary)
                .on_press(Message::ClearCommitted),
        ]
        .spacing(10);

        let mono = |s: String| text(s).font(Font::MONOSPACE).size(13);
        let header = row![
            text("#").width(28),
            text("word").width(Length::Fill),
            text("score").width(70),
            text("edit").width(50),
            text("roman").width(55),
            text("freq").width(80),
        ]
        .spacing(8);
        let rows = self.suggestions.iter().enumerate().fold(
            Column::new().spacing(2),
            |col, (i, s)| {
                let line = row![
                    mono(format!("{}", i + 1)).width(28),
                    bn(s.word.clone()).size(22).width(Length::Fill),
                    mono(format!("{:.3}", s.score)).width(70),
                    mono(cost(Some(s.edit_cost))).width(50),
                    mono(cost(s.roman_cost)).width(55),
                    mono(s.frequency.to_string()).width(80),
                ]
                .spacing(8)
                .align_y(iced::Alignment::Center);
                col.push(
                    button(line)
                        .style(button::text)
                        .width(Length::Fill)
                        .on_press(Message::Pick(i)),
                )
            },
        );

        let completions = bn(if self.completions.is_empty() {
            "—".to_string()
        } else {
            self.completions.join("  ·  ")
        })
        .size(18);

        let mut col = column![
            modes,
            input,
            row![text("Preedit:").size(14), preedit]
                .spacing(10)
                .align_y(iced::Alignment::Center),
            committed,
            horizontal_rule(1),
            text(format!(
                "Suggestions — suggest_input ({})",
                self.suggestions.len()
            ))
            .size(16),
            header,
            scrollable(rows).height(Length::FillPortion(3)),
            horizontal_rule(1),
            text(format!(
                "Prefix completions — complete_prefix ({})",
                self.completions.len()
            ))
            .size(16),
            scrollable(completions).height(Length::FillPortion(1)),
        ]
        .spacing(10);

        if self.font_family.is_none() {
            col = col.push(warning(
                "No Bengali font found: Bangla will not render. Set UPODESH_FONT and \
                 UPODESH_FONT_FAMILY.",
            ));
        }
        if let Some(err) = &self.load_error {
            col = col.push(warning(format!("Embedded index failed to load: {err}")));
        }
        col.into()
    }

    fn view_config(&self) -> Column<'_, Message> {
        let k = &self.knobs;
        let c = &k.config;
        column![
            heading("Ranking (CorrectionConfig)"),
            labeled(
                "max_edit_cost",
                c.max_edit_cost.to_string(),
                slider(0..=400, c.max_edit_cost, knob(Knob::MaxEditCost)).step(5u16),
            ),
            labeled(
                "result limit",
                c.max_results.to_string(),
                slider(1u16..=50, c.max_results as u16, |v| {
                    Message::Knob(Knob::Limit(v.into()))
                }),
            ),
            labeled(
                "frequency_weight",
                format!("{:.3}", c.frequency_weight),
                slider(0.0..=0.2, c.frequency_weight, knob(Knob::FrequencyWeight)).step(0.005),
            ),
            labeled(
                "language_model_weight (no LM loaded)",
                format!("{:.2}", c.language_model_weight),
                slider(
                    0.0..=1.0,
                    c.language_model_weight,
                    knob(Knob::LanguageModelWeight)
                )
                .step(0.01),
            ),
            labeled(
                "roman_weight",
                format!("{:.2}", c.roman_weight),
                slider(0.0..=3.0, c.roman_weight, knob(Knob::RomanWeight)).step(0.05),
            ),
            labeled(
                "max_roman_cost",
                c.max_roman_cost.to_string(),
                slider(0..=400, c.max_roman_cost, knob(Knob::MaxRomanCost)).step(5u16),
            ),
            horizontal_rule(1),
            heading("Edit costs (EditCosts)"),
            labeled(
                "insertion",
                k.insertion.to_string(),
                slider(1..=200, k.insertion, knob(Knob::Insertion)).step(5u16),
            ),
            labeled(
                "deletion",
                k.deletion.to_string(),
                slider(1..=200, k.deletion, knob(Knob::Deletion)).step(5u16),
            ),
            labeled(
                "substitution",
                k.substitution.to_string(),
                slider(1..=200, k.substitution, knob(Knob::Substitution)).step(5u16),
            ),
            labeled(
                "transposition",
                k.transposition.to_string(),
                slider(1..=200, k.transposition, knob(Knob::Transposition)).step(5u16),
            ),
            checkbox("Apply class weights (data/weights.json)", k.class_weights)
                .on_toggle(knob(Knob::ClassWeights)),
            bn("Bangla confusion pairs (শ/ষ/স, ন/ণ, ই/ঈ, ি/ী …) are always kept.").size(12),
            button("Reset defaults")
                .style(button::secondary)
                .on_press(Message::ResetDefaults),
        ]
        .spacing(12)
        .padding([0, 12])
    }

    fn view_perf(&self) -> Column<'_, Message> {
        let s = &self.stats;
        let mut col = column![
            heading("Latency per keystroke"),
            latency_row("okkhor convert", &self.convert),
            latency_row("suggest_input", &self.suggest),
            latency_row("complete_prefix", &self.prefix),
            text(format!(
                "p50 / p95 / p99 / max over the last {WINDOW} keystrokes"
            ))
            .size(11),
            button(text(format!("Benchmark ×{BENCH_RUNS}")))
                .on_press_maybe((!self.bangla.is_empty()).then_some(Message::Benchmark)),
        ]
        .spacing(10)
        .padding([0, 12]);

        if let Some(bench) = &self.bench {
            col = col
                .push(bn(format!("{BENCH_RUNS} runs of {}", bench.query)).size(13))
                .push(percentile_row("suggest_input", fmt_percentiles(&bench.suggest)))
                .push(percentile_row("complete_prefix", fmt_percentiles(&bench.prefix)));
        }

        col.push(horizontal_rule(1))
            .push(heading("Memory (heap, via PeakAlloc)"))
            .push(stat_line("index heap", fmt_bytes(self.index_bytes)))
            .push(stat_line(
                "index load time",
                fmt_duration(self.load_time),
            ))
            .push(stat_line(
                "last query peak",
                fmt_bytes(self.query_peak),
            ))
            .push(stat_line(
                "live after query (incl. results)",
                fmt_signed_bytes(self.query_live),
            ))
            .push(stat_line(
                "total live heap",
                fmt_bytes(PEAK_ALLOC.current_usage()),
            ))
            .push(
                text(
                    "Per-query figures are approximate: iced's render threads share \
                     the global allocator.",
                )
                .size(11),
            )
            .push(horizontal_rule(1))
            .push(heading("Index (IndexStats)"))
            .push(stat_line("words", s.words.to_string()))
            .push(stat_line("graphemes", s.graphemes.to_string()))
            .push(stat_line("max word length", s.max_dict_len.to_string()))
            .push(stat_line("max frequency", s.max_frequency.to_string()))
    }
}

/// Wrap a `Knob` constructor as a widget callback.
fn knob<T>(make: fn(T) -> Knob) -> impl Fn(T) -> Message {
    move |v| Message::Knob(make(v))
}

/// Text that may contain Bangla: needs full shaping for conjuncts and
/// pre-base vowel signs (iced's default `Shaping::Basic` mangles them).
fn bn<'a>(content: impl text::IntoFragment<'a>) -> text::Text<'a, Theme> {
    text(content).shaping(text::Shaping::Advanced)
}

fn heading<'a>(label: &'a str) -> Element<'a, Message> {
    text(label).size(17).into()
}

fn warning<'a>(message: impl text::IntoFragment<'a>) -> Element<'a, Message> {
    text(message)
        .size(13)
        .style(|theme: &Theme| text::Style {
            color: Some(theme.palette().danger),
        })
        .into()
}

fn labeled<'a>(
    label: &'a str,
    value: String,
    control: impl Into<Element<'a, Message>>,
) -> Element<'a, Message> {
    column![
        row![
            text(label).size(13).width(Length::Fill),
            text(value).font(Font::MONOSPACE).size(13),
        ],
        control.into(),
    ]
    .spacing(4)
    .into()
}

fn stat_line<'a>(label: &'a str, value: String) -> Element<'a, Message> {
    row![
        text(label).size(13).width(Length::Fill),
        text(value).font(Font::MONOSPACE).size(13),
    ]
    .spacing(8)
    .into()
}

fn latency_row<'a>(label: &'a str, samples: &Samples) -> Element<'a, Message> {
    let summary = percentiles(samples.window.iter().copied())
        .map(|p| fmt_percentiles(&p))
        .unwrap_or_else(|| "—".to_string());
    column![
        stat_line(label, format!("last {}", fmt_duration(samples.last))),
        text(summary).font(Font::MONOSPACE).size(12),
    ]
    .spacing(2)
    .into()
}

fn percentile_row<'a>(label: &'a str, summary: String) -> Element<'a, Message> {
    column![
        text(label).size(13),
        text(summary).font(Font::MONOSPACE).size(12),
    ]
    .spacing(2)
    .into()
}

fn cost(cost: Option<u16>) -> String {
    match cost {
        None => "—".to_string(),
        Some(u16::MAX) => "∞".to_string(),
        Some(c) => c.to_string(),
    }
}

fn fmt_duration(d: Duration) -> String {
    let us = d.as_secs_f64() * 1e6;
    if us >= 1000.0 {
        format!("{:.2} ms", us / 1000.0)
    } else {
        format!("{us:.1} µs")
    }
}

fn fmt_percentiles(p: &[Duration; 4]) -> String {
    p.iter()
        .map(|d| fmt_duration(*d))
        .collect::<Vec<_>>()
        .join(" / ")
}

fn fmt_bytes(bytes: usize) -> String {
    let b = bytes as f64;
    if b >= 1024.0 * 1024.0 {
        format!("{:.2} MB", b / (1024.0 * 1024.0))
    } else if b >= 1024.0 {
        format!("{:.1} KB", b / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

fn fmt_signed_bytes(bytes: isize) -> String {
    let sign = if bytes < 0 { "-" } else { "+" };
    format!("{sign}{}", fmt_bytes(bytes.unsigned_abs()))
}
