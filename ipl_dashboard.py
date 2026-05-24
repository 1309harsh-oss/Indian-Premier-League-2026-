"""
IPL Advanced Analytics Dashboard — Python / Matplotlib
Data: cricsheet.org  (2007–2026 · 1,218 matches · 289,673 deliveries)
Run : python ipl_dashboard_v2.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch, Wedge
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH = "att_0_1778303821_c3a907.csv"

# Palette
BG      = "#04050d"
SURFACE = "#0c0f1e"
CARD    = "#0f1322"
CARD2   = "#131828"
BORDER  = "#1e2540"
GOLD    = "#f5a623"
GOLD2   = "#ffcc55"
BLUE    = "#00c9ff"
RED     = "#ff4d6d"
GREEN   = "#39d353"
PURPLE  = "#c084fc"
TEAL    = "#2dd4bf"
MUTED   = "#4b5563"
TEXT    = "#e8eaf6"
TEXTD   = "#9ca3af"

def hex_alpha(h, a):
    r,g,b = mcolors.hex2color(h)
    return (r,g,b,a)

matplotlib.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    CARD,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXTD,
    "xtick.color":       TEXTD,
    "ytick.color":       TEXTD,
    "text.color":        TEXT,
    "grid.color":        BORDER,
    "grid.linewidth":    0.5,
    "axes.grid":         True,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.titlepad":     10,
})

# ─────────────────────────────────────────────────────────────────────────────
# LOAD & PREPARE
# ─────────────────────────────────────────────────────────────────────────────
print("⏳  Loading …")
df = pd.read_csv(CSV_PATH, low_memory=False,
                 dtype={"wicket_kind": str, "wicket_player_out": str})
df['season'] = df['season'].astype(str)
df['winner'] = df['winner'].fillna("")
print(f"✅  {len(df):,} balls  ·  {df['match_id'].nunique():,} matches")

# ── Match-level meta ──────────────────────────────────────────────────────────
meta = (df.drop_duplicates("match_id")
          [["match_id","season","toss_winner","toss_decision","winner","team1","team2"]]
          .copy())
valid = meta[meta["winner"] != ""].copy()
valid["toss_won_match"] = valid["toss_winner"] == valid["winner"]

N = len(valid)

# ── Toss ─────────────────────────────────────────────────────────────────────
toss_yes = valid["toss_won_match"].sum()
toss_no  = N - toss_yes

# ── Season toss win % (keep unique seasons, sorted) ──────────────────────────
season_order = ["2007/08","2009","2009/10","2011","2012","2013","2014","2015",
                "2016","2017","2018","2019","2019","2020/21","2021","2022",
                "2023","2024","2025","2026"]
s_toss = (valid.groupby("season")["toss_won_match"]
                .mean()
                .mul(100)
                .reset_index()
                .rename(columns={"toss_won_match":"pct"}))
# deduplicate + sort properly
seen = {}
for _, row in s_toss.iterrows():
    seen[row["season"]] = row["pct"]
unique_seasons = []
for s in season_order:
    if s in seen and s not in unique_seasons:
        unique_seasons.append(s)
for s in s_toss["season"]:
    if s not in unique_seasons:
        unique_seasons.append(s)
s_toss_clean = pd.DataFrame({"season": unique_seasons,
                              "pct": [seen[s] for s in unique_seasons]})

# ── Phase run rates ───────────────────────────────────────────────────────────
def phase(o):
    if o <= 5:  return "Powerplay"
    if o <= 14: return "Middle"
    return "Death"

df["phase"] = df["over"].apply(phase)
df["winner_batting"] = df["batting_team"] == df["winner"]

pa = (df.groupby(["phase","winner_batting"])["runs_total"]
        .agg(s="sum", n="count"))
pa["rpo"] = pa["s"] / pa["n"] * 6
PHASES = ["Powerplay","Middle","Death"]
win_rpo  = [pa.loc[(p, True),  "rpo"] for p in PHASES]
loss_rpo = [pa.loc[(p, False), "rpo"] for p in PHASES]

# ── Top batters / bowlers ─────────────────────────────────────────────────────
top_bat  = df.groupby("batter")["runs_batter"].sum().nlargest(8).reset_index()
top_bat.columns = ["Player","Runs"]

VALID_WK = {"caught","bowled","lbw","stumped","caught and bowled","hit wicket"}
top_bowl = (df[df["wicket_kind"].isin(VALID_WK)]
              .groupby("bowler")["wicket_kind"].count()
              .nlargest(8).reset_index())
top_bowl.columns = ["Player","Wickets"]

# ── Team wins ─────────────────────────────────────────────────────────────────
team_wins = valid["winner"].value_counts().head(8)

# ── Season avg run rate (total runs / total balls * 6) ───────────────────────
season_rr = (df.groupby("season")
               .apply(lambda g: g["runs_total"].sum() / len(g) * 6)
               .reset_index())
season_rr.columns = ["season","rr"]
season_rr = season_rr[season_rr["season"].isin(unique_seasons)]
season_rr["ord"] = season_rr["season"].map({s:i for i,s in enumerate(unique_seasons)})
season_rr = season_rr.sort_values("ord").reset_index(drop=True)

# ── Sixes leaders ─────────────────────────────────────────────────────────────
sixes = df[df["runs_batter"]==6].groupby("batter").size().nlargest(6).reset_index()
sixes.columns = ["Player","Sixes"]

# ── Strike rate leaders (min 300 balls) ──────────────────────────────────────
balls_faced = df.groupby("batter").size()
runs_scored = df.groupby("batter")["runs_batter"].sum()
sr = (runs_scored / balls_faced * 100).where(balls_faced >= 300).dropna().nlargest(6).reset_index()
sr.columns = ["Player","SR"]

# ── Toss decision breakdown ───────────────────────────────────────────────────
dec = valid["toss_decision"].value_counts()

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE  —  4 rows × 3 cols  +  footer
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 26))
fig.patch.set_facecolor(BG)

gs = GridSpec(5, 3, figure=fig,
              hspace=0.55, wspace=0.35,
              top=0.93, bottom=0.03,
              left=0.055, right=0.975,
              height_ratios=[1, 1, 1, 1, 0.55])

def make_ax(row, col, colspan=1):
    if colspan > 1:
        return fig.add_subplot(gs[row, col:col+colspan])
    return fig.add_subplot(gs[row, col])

ax_toss_bar   = make_ax(0, 0)          # toss win/loss bar
ax_toss_trend = make_ax(0, 1)          # season toss win %
ax_toss_pie   = make_ax(0, 2)          # toss decision pie
ax_phase      = make_ax(1, 0, 2)       # phase run rates (wide)
ax_rr         = make_ax(1, 2)          # season run rate trend
ax_bat        = make_ax(2, 0)          # top batters
ax_bowl       = make_ax(2, 1)          # top bowlers
ax_team       = make_ax(2, 2)          # team wins
ax_six        = make_ax(3, 0)          # sixes leaders
ax_sr         = make_ax(3, 1)          # strike rate leaders
ax_insight    = make_ax(3, 2)          # insight box
ax_footer     = make_ax(4, 0, 3)       # surprise finding


def style(ax, title, tc=BLUE, grid_axis="y"):
    ax.set_facecolor(CARD)
    for sp in ax.spines.values(): sp.set_color(BORDER); sp.set_linewidth(0.7)
    ax.set_title(title, color=tc, fontsize=10, fontweight="bold",
                 loc="left", pad=9)
    ax.tick_params(colors=TEXTD, labelsize=8.5, length=3)
    ax.grid(axis=grid_axis, color=BORDER, linewidth=0.5, zorder=0)
    ax.grid(axis="x" if grid_axis == "y" else "y", visible=False)


# ──────────────────────────────────────────────────────────────────────────────
# 1. TOSS WIN / LOSS BAR
# ──────────────────────────────────────────────────────────────────────────────
style(ax_toss_bar, "🎲 Toss vs Match Result", GOLD)

cats = ["Toss Winner\nWon Match", "Toss Winner\nLost Match"]
vals = [toss_yes, toss_no]
cols = [GOLD, RED]

bars = ax_toss_bar.bar(cats, vals, color=cols, alpha=0.87,
                       width=0.45, zorder=3,
                       edgecolor=[hex_alpha(GOLD,0.4), hex_alpha(RED,0.4)],
                       linewidth=0.8)

for bar, v in zip(bars, vals):
    pct = v / N * 100
    ax_toss_bar.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 8,
                     f"{v:,}\n({pct:.1f}%)",
                     ha="center", va="bottom",
                     color=TEXT, fontsize=9, fontweight="bold")

ax_toss_bar.set_ylim(0, max(vals) * 1.28)
ax_toss_bar.set_xticklabels(cats, color=TEXT, fontsize=9)
ax_toss_bar.set_ylabel("Number of Matches", fontsize=8.5)

# horizontal 50% reference line
ax_toss_bar.axhline(N/2, color=MUTED, linewidth=0.9, linestyle="--", zorder=2)
ax_toss_bar.text(1.5, N/2 + 5, "50%", color=MUTED, fontsize=7.5, ha="right")


# ──────────────────────────────────────────────────────────────────────────────
# 2. SEASON TOSS WIN % TREND
# ──────────────────────────────────────────────────────────────────────────────
style(ax_toss_trend, "📈 Toss Win % by Season", BLUE)

x = range(len(s_toss_clean))
y = s_toss_clean["pct"].values

ax_toss_trend.plot(x, y, color=BLUE, linewidth=1.6, zorder=4)
ax_toss_trend.fill_between(x, y, 50, where=(y >= 50),
                            color=GREEN, alpha=0.18, zorder=2)
ax_toss_trend.fill_between(x, y, 50, where=(y < 50),
                            color=RED, alpha=0.18, zorder=2)
ax_toss_trend.scatter(x, y, color=BLUE, s=30, zorder=5, edgecolors=BG, linewidths=0.6)
ax_toss_trend.axhline(50, color=MUTED, linewidth=1, linestyle="--", zorder=3)

ax_toss_trend.set_xticks(list(x))
ax_toss_trend.set_xticklabels(s_toss_clean["season"], rotation=55, ha="right", fontsize=7)
ax_toss_trend.set_ylabel("Win %", fontsize=8.5)
ax_toss_trend.set_ylim(25, 80)
ax_toss_trend.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))


# ──────────────────────────────────────────────────────────────────────────────
# 3. TOSS DECISION PIE
# ──────────────────────────────────────────────────────────────────────────────
ax_toss_pie.set_facecolor(CARD)
ax_toss_pie.set_title("🏏 Toss Decision Split", color=PURPLE,
                      fontsize=10, fontweight="bold", loc="left", pad=9)

sizes  = [dec.get("field", 0), dec.get("bat", 0)]
labels = [f"Field First\n{dec.get('field',0):,}", f"Bat First\n{dec.get('bat',0):,}"]
pcolors = [BLUE, GOLD]
explode = [0.04, 0.04]

wedges, texts, autotexts = ax_toss_pie.pie(
    sizes, labels=None, colors=pcolors, explode=explode,
    autopct="%1.1f%%", startangle=90,
    wedgeprops={"edgecolor": BG, "linewidth": 2},
    textprops={"color": TEXT, "fontsize": 9},
    pctdistance=0.72,
)
for at in autotexts: at.set_fontsize(9); at.set_fontweight("bold")
ax_toss_pie.legend(wedges, labels, loc="lower center",
                   bbox_to_anchor=(0.5, -0.12), ncol=2,
                   facecolor=SURFACE, edgecolor=BORDER,
                   labelcolor=TEXT, fontsize=8)


# ──────────────────────────────────────────────────────────────────────────────
# 4. PHASE RUN RATES  (grouped bars + difference line)
# ──────────────────────────────────────────────────────────────────────────────
style(ax_phase, "📊 Average Runs / Over by Phase  —  Winners vs Losers", GOLD)

x3 = np.arange(3)
bw = 0.3

b1 = ax_phase.bar(x3 - bw/2, win_rpo,  width=bw, color=GOLD, alpha=0.88,
                  zorder=3, label="Winners", edgecolor=hex_alpha(GOLD,0.5), linewidth=0.6)
b2 = ax_phase.bar(x3 + bw/2, loss_rpo, width=bw, color=RED,  alpha=0.80,
                  zorder=3, label="Losers",  edgecolor=hex_alpha(RED,0.5),  linewidth=0.6)

for bar, v in zip(b1, win_rpo):
    ax_phase.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.07,
                  f"{v:.2f}", ha="center", va="bottom",
                  color=GOLD, fontsize=9, fontweight="bold")
for bar, v in zip(b2, loss_rpo):
    ax_phase.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.07,
                  f"{v:.2f}", ha="center", va="bottom",
                  color=RED, fontsize=9)

# gap annotations
for i, (w, l) in enumerate(zip(win_rpo, loss_rpo)):
    g = w - l
    mid_y = max(w, l) + 0.52
    ax_phase.annotate(f"Δ +{g:.2f}", xy=(i, mid_y),
                      ha="center", fontsize=8.5, color=GREEN, fontweight="bold")

ax_phase.set_xticks(x3)
ax_phase.set_xticklabels(
    ["Powerplay  (Overs 1–6)", "Middle Overs  (7–15)", "Death Overs  (16–20)"],
    fontsize=10, color=TEXT)
ax_phase.set_ylabel("Avg Runs / Over", fontsize=9)
ax_phase.set_ylim(0, max(win_rpo) * 1.38)
ax_phase.legend(facecolor=SURFACE, edgecolor=BORDER,
                labelcolor=TEXT, fontsize=9, loc="upper left")


# ──────────────────────────────────────────────────────────────────────────────
# 5. SEASON RUN RATE TREND
# ──────────────────────────────────────────────────────────────────────────────
style(ax_rr, "⚡ IPL Run Rate Over Seasons", TEAL)

xr = np.arange(len(season_rr))
yr = season_rr["rr"].values

# colour gradient segments
for i in range(len(xr)-1):
    t = i / max(len(xr)-2, 1)
    c = mcolors.to_hex(plt.cm.cool(t))
    ax_rr.plot(xr[i:i+2], yr[i:i+2], color=c, linewidth=2, zorder=4)

ax_rr.fill_between(xr, yr, yr.min()-0.1, alpha=0.12, color=TEAL, zorder=2)
ax_rr.scatter(xr, yr, color=TEAL, s=30, zorder=5, edgecolors=BG, linewidths=0.6)

# label max
idx_max = np.argmax(yr)
ax_rr.annotate(f"{yr[idx_max]:.2f}", xy=(xr[idx_max], yr[idx_max]),
               xytext=(xr[idx_max]-0.5, yr[idx_max]+0.12),
               color=GREEN, fontsize=8, fontweight="bold")

ax_rr.set_xticks(xr)
ax_rr.set_xticklabels(season_rr["season"], rotation=55, ha="right", fontsize=6.8)
ax_rr.set_ylabel("Avg Runs / Over", fontsize=8.5)


# ──────────────────────────────────────────────────────────────────────────────
# HELPER — horizontal bar chart
# ──────────────────────────────────────────────────────────────────────────────
MEDALS = ["🥇","🥈","🥉","4.","5.","6.","7.","8."]

def hbar(ax, players, values, title, tc, bar_color, unit=""):
    style(ax, title, tc, grid_axis="x")
    y = np.arange(len(players))
    bars = ax.barh(y, values, height=0.55, color=bar_color,
                   alpha=0.85, zorder=3,
                   edgecolor=hex_alpha(bar_color if isinstance(bar_color,str) else bar_color[0], 0.4),
                   linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(players, fontsize=9, color=TEXT)
    ax.invert_yaxis()
    ax.set_xlabel(unit, fontsize=8.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + max(values)*0.01,
                bar.get_y() + bar.get_height()/2,
                f"{int(v):,}" if isinstance(v, (int, np.integer)) or v == int(v) else f"{v:.1f}",
                va="center", color=TEXT, fontsize=8.5, fontweight="bold")
    ax.set_xlim(0, max(values) * 1.18)


# ──────────────────────────────────────────────────────────────────────────────
# 6. TOP BATTERS
# ──────────────────────────────────────────────────────────────────────────────
hbar(ax_bat,
     top_bat["Player"].tolist(),
     top_bat["Runs"].tolist(),
     "🏏 Top 8 Batters — Total Runs", GOLD,
     GOLD, unit="Runs Scored")


# ──────────────────────────────────────────────────────────────────────────────
# 7. TOP BOWLERS
# ──────────────────────────────────────────────────────────────────────────────
hbar(ax_bowl,
     top_bowl["Player"].tolist(),
     top_bowl["Wickets"].tolist(),
     "🎯 Top 8 Bowlers — Wickets", BLUE,
     BLUE, unit="Wickets Taken")


# ──────────────────────────────────────────────────────────────────────────────
# 8. TEAM WIN COUNTS
# ──────────────────────────────────────────────────────────────────────────────
style(ax_team, "🏆 Most Wins — All Time", PURPLE, grid_axis="x")

teams  = team_wins.index.tolist()
wins_v = team_wins.values.tolist()
y_t    = np.arange(len(teams))

team_colors = [GOLD, GOLD2, TEAL, BLUE, RED, GREEN, PURPLE, "#f97316"][:len(teams)]

bars_t = ax_team.barh(y_t, wins_v, height=0.55, color=team_colors,
                      alpha=0.85, zorder=3)
ax_team.set_yticks(y_t)
ax_team.set_yticklabels([t.replace(" ","  ") for t in teams], fontsize=8, color=TEXT)
ax_team.invert_yaxis()
ax_team.set_xlabel("Matches Won", fontsize=8.5)
for bar, v in zip(bars_t, wins_v):
    ax_team.text(bar.get_width() + 1.5,
                 bar.get_y() + bar.get_height()/2,
                 str(v), va="center", color=TEXT, fontsize=8.5, fontweight="bold")
ax_team.set_xlim(0, max(wins_v) * 1.2)


# ──────────────────────────────────────────────────────────────────────────────
# 9. SIXES LEADERS
# ──────────────────────────────────────────────────────────────────────────────
hbar(ax_six,
     sixes["Player"].tolist(),
     sixes["Sixes"].tolist(),
     "💥 Six-Hitting Kings", RED,
     RED, unit="Total Sixes")


# ──────────────────────────────────────────────────────────────────────────────
# 10. STRIKE RATE LEADERS
# ──────────────────────────────────────────────────────────────────────────────
hbar(ax_sr,
     sr["Player"].tolist(),
     sr["SR"].tolist(),
     "⚡ Best Strike Rates  (min 300 balls)", GREEN,
     GREEN, unit="Strike Rate")


# ──────────────────────────────────────────────────────────────────────────────
# 11. INSIGHT SUMMARY BOX  (top-right)
# ──────────────────────────────────────────────────────────────────────────────
ax_insight.set_facecolor(CARD)
ax_insight.set_xlim(0, 1); ax_insight.set_ylim(0, 1)
ax_insight.axis("off")

# Card border
rect = FancyBboxPatch((0.03, 0.04), 0.94, 0.92,
                      boxstyle="round,pad=0.02",
                      linewidth=1.4, edgecolor=GOLD,
                      facecolor="#13111f", zorder=1)
ax_insight.add_patch(rect)

ax_insight.text(0.09, 0.87, "📌  Key Numbers", transform=ax_insight.transAxes,
                color=GOLD, fontsize=10, fontweight="bold", va="top", zorder=2)

stats = [
    ("1,218",   "Total Matches"),
    ("289,673", "Total Deliveries"),
    ("9,050",   "Kohli's Total Runs"),
    ("229",     "Chahal's Wickets"),
    ("50.5%",   "Toss → Win Rate"),
    ("+1.78",   "Death Over RPO Gap"),
    ("66%",     "Teams Field First"),
    ("154",     "Mumbai Indians Wins"),
]
for i, (val, lbl) in enumerate(stats):
    col_x = 0.09 + (i % 2) * 0.5
    row_y = 0.74 - (i // 2) * 0.17
    ax_insight.text(col_x, row_y, val, transform=ax_insight.transAxes,
                    color=BLUE if i % 2 == 0 else GOLD,
                    fontsize=11, fontweight="bold", va="top", zorder=2)
    ax_insight.text(col_x, row_y - 0.07, lbl, transform=ax_insight.transAxes,
                    color=TEXTD, fontsize=7.5, va="top", zorder=2)


# ──────────────────────────────────────────────────────────────────────────────
# 12. SURPRISE FINDING FOOTER
# ──────────────────────────────────────────────────────────────────────────────
ax_footer.set_facecolor(CARD)
ax_footer.set_xlim(0, 1); ax_footer.set_ylim(0, 1)
ax_footer.axis("off")

rect2 = FancyBboxPatch((0.005, 0.05), 0.99, 0.88,
                       boxstyle="round,pad=0.01",
                       linewidth=1.2, edgecolor=GREEN,
                       facecolor="#0a1a12", zorder=1)
ax_footer.add_patch(rect2)

ax_footer.text(0.015, 0.82, "💡  Surprise Finding —",
               transform=ax_footer.transAxes,
               color=GREEN, fontsize=10, fontweight="bold", va="top", zorder=2)

finding = (
    "Winning the toss is almost meaningless: toss winners win only 50.5% of matches — virtually a coin flip. "
    "Yet teams that dominate the Death Overs (16–20) show a decisive +1.78 runs/over advantage over losing sides "
    "(10.51 vs 8.73), far larger than the Powerplay gap (+0.94) or Middle-Overs gap (+0.95). "
    "Additionally, 66% of toss-winning captains chose to field first — yet that strategy produces no measurable win-rate edge, "
    "suggesting the dew-factor logic is widely over-rated in the data."
)
ax_footer.text(0.015, 0.54, finding,
               transform=ax_footer.transAxes,
               color=TEXTD, fontsize=9.2, va="top", linespacing=1.6, zorder=2,
               wrap=True)


# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
fig.text(0.055, 0.965, "🏏  IPL Analytics — Advanced Dashboard",
         fontsize=20, fontweight="bold", color=TEXT, va="top")
fig.text(0.055, 0.950,
         "Seasons 2007 – 2026   ·   1,218 Matches   ·   289,673 Deliveries   ·   Source: cricsheet.org",
         fontsize=9, color=MUTED, va="top")

# Thin gold separator line
fig.add_artist(plt.Line2D([0.055, 0.975], [0.940, 0.940],
                          transform=fig.transFigure,
                          color=GOLD, linewidth=0.8, alpha=0.5))

# ──────────────────────────────────────────────────────────────────────────────
# SAVE
# ──────────────────────────────────────────────────────────────────────────────
OUT = "ipl_advanced_dashboard.png"
plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor=BG)
print(f"\n✅  Saved → {OUT}")
