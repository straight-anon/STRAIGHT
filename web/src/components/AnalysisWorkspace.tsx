import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import styled from 'styled-components';
import ReactMarkdown from 'react-markdown';
import { type PhaseSegment } from '../services/api';
import logoBadge from '../assets/tpn_logo.png';

export type GhostTiming = {
  drawSeconds?: number;
  avgSeconds?: number;
  stdSeconds?: number;
  status?: string;
  sampleCount?: number;
  minSamples?: number;
};

export type SpineSummary = {
  normalizedDistance?: number;
  status?: string;
};

export type DrawForceSummary = {
  angleDeg?: number;
  angleAvg?: number;
  angleStd?: number;
  angleSampleCount?: number;
  drawLengthSampleCount?: number;
  minSamplesRequired?: number;
  drawLengthHw?: number;
  drawLengthAvg?: number;
  drawLengthStd?: number;
  status?: string;
};

export type PostReleaseDrawSummary = {
  prePx?: number;
  postPx?: number;
  changePct?: number;
  thresholdPct?: number;
  status?: string;
};

export type PostReleaseBowSummary = {
  preDeg?: number;
  postDeg?: number;
  deltaDeg?: number;
  thresholdDeg?: number;
  status?: string;
};

export type AnalysisTab = {
  id: string;
  label: string;
  markdown: string;
  frames?: string[];
  ghostTiming?: GhostTiming;
  image?: string;
  spineSummary?: SpineSummary;
  drawForceSummary?: DrawForceSummary;
  postReleaseDrawSummary?: PostReleaseDrawSummary;
  postReleaseBowSummary?: PostReleaseBowSummary;
};

export type VideoSources = {
  original?: string;
  skeleton?: string;
  ghost?: string;
};

const MIN_DATASET_SAMPLES = 5;

const Page = styled.div`
  min-height: 100vh;
  padding: ${({ theme }) => theme.spacing(3)};
  background: radial-gradient(circle at top left, rgba(140, 103, 255, 0.12), transparent 40%),
    ${({ theme }) => theme.colors.background};
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing(2)};
`;

const HeaderBar = styled.header`
  max-width: 1180px;
  width: 100%;
  margin: 0 auto;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: ${({ theme }) => theme.spacing(2)};
  border-radius: ${({ theme }) => theme.radii.md};
  background: ${({ theme }) => theme.colors.surface};
  box-shadow: ${({ theme }) => theme.shadows.glow};
  border: 1px solid rgba(0, 0, 0, 0.04);
`;

const BrandGroup = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing(1.5)};
`;

const LogoBadge = styled.img`
  height: 40px;
  width: auto;
  display: block;
`;

const BrandText = styled.span`
  font-weight: 700;
  letter-spacing: 0.03em;
  font-family: ${({ theme }) => theme.fonts.heading};
`;

const NavActions = styled.nav`
  display: flex;
  gap: ${({ theme }) => theme.spacing(1)};
  align-items: center;
`;

const IconButton = styled.a`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 42px;
  height: 42px;
  border-radius: 12px;
  background: ${({ theme }) => theme.colors.surfaceAlt};
  color: ${({ theme }) => theme.colors.text};
  text-decoration: none;
  border: 1px solid rgba(0, 0, 0, 0.04);
  transition: transform ${({ theme }) => theme.durations.quick} ease, box-shadow ${({ theme }) =>
      theme.durations.quick} ease;

  &:hover,
  &:focus-visible {
    transform: translateY(-2px);
    box-shadow: ${({ theme }) => theme.shadows.glow};
    outline: none;
  }
`;

const IconSvg = styled.svg`
  width: 22px;
  height: 22px;
  fill: currentColor;
`;

const Shell = styled.section`
  height: calc(100vh - ${({ theme }) => theme.spacing(10)});
  max-width: 1180px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: minmax(320px, 55%) minmax(260px, 1fr);
  gap: ${({ theme }) => theme.spacing(3)};
  padding: ${({ theme }) => theme.spacing(2)};
  align-items: stretch;
  overflow: hidden;
`;

const VideoPanel = styled.div`
  background: ${({ theme }) => theme.colors.surface};
  border-radius: ${({ theme }) => theme.radii.lg};
  padding: ${({ theme }) => theme.spacing(2.5)};
  box-shadow: ${({ theme }) => theme.shadows.glow};
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
`;

const VideoSurface = styled.div<{ $aspect: number }>`
  position: relative;
  width: 100%;
  aspect-ratio: ${({ $aspect }) => $aspect};
  border-radius: ${({ theme }) => theme.radii.md};
  overflow: hidden;
  background: linear-gradient(135deg, rgba(70, 248, 194, 0.2), rgba(140, 103, 255, 0.15));
  box-shadow: ${({ theme }) => theme.shadows.glow};
`;

const VideoElement = styled.video`
  width: 100%;
  height: 100%;
  object-fit: contain;
  border-radius: ${({ theme }) => theme.radii.md};
  background: linear-gradient(135deg, rgba(70, 248, 194, 0.1), rgba(140, 103, 255, 0.1));
  border: 1px solid rgba(0, 0, 0, 0.04);
`;

const Controls = styled.div`
  position: absolute;
  left: 12px;
  right: 12px;
  bottom: 12px;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 12px;
  border-radius: 12px;
  background: rgba(0, 0, 0, 0.42);
  color: #fefefe;
  backdrop-filter: blur(6px);
`;

const ControlButton = styled.button`
  border: none;
  background: ${({ theme }) => theme.colors.neon};
  color: #05070f;
  border-radius: 10px;
  padding: 8px 12px;
  font-weight: 700;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);

  &:hover,
  &:focus-visible {
    filter: brightness(1.05);
    outline: none;
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const Seek = styled.input`
  flex: 1;
  appearance: none;
  height: 6px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.4);
  outline: none;
  margin: 0;
  accent-color: ${({ theme }) => theme.colors.neon};

  &::-webkit-slider-thumb {
    appearance: none;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: ${({ theme }) => theme.colors.neon};
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
    cursor: pointer;
  }

  &::-moz-range-thumb {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: ${({ theme }) => theme.colors.neon};
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
    cursor: pointer;
    border: none;
  }
`;

const Timecode = styled.span`
  font-family: 'SFMono-Regular', Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
  font-size: 0.85rem;
  min-width: 92px;
  text-align: right;
  color: #fefefe;
`;

const VideoFallback = styled.div`
  width: 100%;
  height: 100%;
  display: grid;
  place-content: center;
  color: ${({ theme }) => theme.colors.muted};
`;


const TabBar = styled.div`
  display: grid;
  grid-template-columns: 42px 1fr 42px;
  align-items: center;
  gap: ${({ theme }) => theme.spacing(1)};
  margin-bottom: ${({ theme }) => theme.spacing(2)};
  background: ${({ theme }) => theme.colors.surface};
  border-radius: ${({ theme }) => theme.radii.md};
  padding: 6px;
  box-shadow: ${({ theme }) => theme.shadows.glow};
`;

const TabsArrow = styled.button<{ $side: 'left' | 'right' }>`
  background: ${({ theme }) => theme.colors.surfaceAlt};
  color: ${({ theme }) => theme.colors.neonAlt};
  border-radius: 50%;
  width: 30px;
  height: 30px;
  display: grid;
  place-items: center;
  font-weight: 800;
  border: 2px solid ${({ theme }) => theme.colors.neon};
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25), ${({ theme }) => theme.shadows.glow};
  cursor: pointer;
  opacity: ${({ disabled }) => (disabled ? 0.25 : 1)};
  transition: opacity 120ms ease, transform 120ms ease;

  &:hover:not(:disabled) {
    transform: translateY(-1px);
  }

  &:disabled {
    cursor: default;
  }
`;

const TabScrollArea = styled.div`
  position: relative;
  overflow-x: auto;
  overflow-y: hidden;
  white-space: nowrap;
  width: 100%;

  &::-webkit-scrollbar {
    display: none;
  }
`;

const Tabs = styled.nav`
  display: inline-flex;
  gap: ${({ theme }) => theme.spacing(2)};
  padding: ${({ theme }) => `${theme.spacing(0.5)} ${theme.spacing(1)}`};
  white-space: nowrap;
`;

const TabButton = styled.button<{ $active: boolean }>`
  flex: 0 0 auto;
  padding: ${({ theme }) => theme.spacing(1)};
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid
    ${({ $active, theme }) => ($active ? theme.colors.neon : 'rgba(255,255,255,0.2)')};
  background: ${({ $active, theme }) => ($active ? theme.colors.surfaceAlt : 'transparent')};
  color: ${({ theme }) => theme.colors.text};
  cursor: pointer;
`;

const ToggleGroup = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing(1)};
  margin: ${({ theme }) => theme.spacing(2)} 0;
`;

const ToggleButton = styled.button<{ $active: boolean }>`
  flex: 1;
  padding: ${({ theme }) => theme.spacing(1)};
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid
    ${({ $active, theme }) => ($active ? theme.colors.neon : 'rgba(255,255,255,0.2)')};
  background: ${({ $active, theme }) => ($active ? theme.colors.surfaceAlt : 'transparent')};
  color: ${({ theme }) => theme.colors.text};
  cursor: pointer;
`;

const ProgressShell = styled.div<{ $compact?: boolean }>`
  margin-top: ${({ theme, $compact }) => ($compact ? theme.spacing(1) : theme.spacing(2))};
`;

const ProgressHeader = styled.div<{ $compact?: boolean }>`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: ${({ $compact }) => ($compact ? 6 : 8)}px;
  color: ${({ theme }) => theme.colors.muted};
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-size: 0.75rem;
`;

const ProgressBar = styled.div`
  position: relative;
  height: 10px;
  border-radius: 999px;
  overflow: visible;
  display: flex;
  background: rgba(255, 255, 255, 0.08);
`;

const ProgressSegment = styled.div<{ $color: string; $width: number }>`
  background: ${({ $color }) => $color};
  width: ${({ $width }) => `${$width}%`};
  min-width: 2px;
`;

const ProgressPointer = styled.span<{ $position?: number }>`
  position: absolute;
  top: -10px;
  min-width: 22px;
  height: 22px;
  border-radius: 10px;
  background: ${({ theme }) => theme.colors.surface};
  border: 2px solid ${({ theme }) => theme.colors.neonAlt};
  color: ${({ theme }) => theme.colors.neonAlt};
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-weight: 800;
  left: ${({ $position }) => ($position === undefined ? '-9999px' : `${$position}%`)};
  transform: translateX(-50%);
  box-shadow: ${({ theme }) => theme.shadows.glow};
  opacity: ${({ $position }) => ($position === undefined ? 0 : 1)};
  transition: left 120ms ease, opacity 120ms ease;
`;

const MarkdownPanel = styled.article`
  background: ${({ theme }) => theme.colors.surface};
  border-radius: ${({ theme }) => theme.radii.lg};
  padding: ${({ theme }) => theme.spacing(2.5)};
  box-shadow: ${({ theme }) => theme.shadows.glow};
  overflow-y: auto;
  max-height: 100%;
  min-height: 0;
`;

const SpineCard = styled.div`
  border: 1px solid rgba(0, 0, 0, 0.05);
  background: ${({ theme }) => theme.colors.surfaceAlt};
  border-radius: ${({ theme }) => theme.radii.md};
  padding: ${({ theme }) => theme.spacing(1.5)};
  margin-bottom: ${({ theme }) => theme.spacing(2)};
  box-shadow: ${({ theme }) => theme.shadows.glow};
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const SpineMetric = styled.div`
  display: flex;
  flex-direction: column;
  gap: 4px;
`;

const SpineLabel = styled.span`
  font-size: 0.9rem;
  color: ${({ theme }) => theme.colors.muted};
`;

const SpineValue = styled.span`
  font-size: 1.15rem;
  font-weight: 700;
  color: ${({ theme }) => theme.colors.text};
`;

const SpineChip = styled.span<{ $tone?: 'warn' }>`
  padding: 6px 10px;
  border-radius: 999px;
  background: ${({ $tone }) => ($tone === 'warn' ? 'rgba(255, 122, 26, 0.2)' : 'rgba(70, 248, 194, 0.18)')};
  color: ${({ theme }) => theme.colors.text};
  border: ${({ $tone }) => ($tone === 'warn' ? '1px solid rgba(255, 122, 26, 0.55)' : '1px solid rgba(70, 248, 194, 0.4)')};
  font-weight: 700;
`;

const DrawForceCard = styled(SpineCard)`
  flex-direction: column;
  align-items: flex-start;
  gap: ${({ theme }) => theme.spacing(1)};
`;

const StatRow = styled.div`
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
`;

const StaticImageShell = styled.div`
  background: ${({ theme }) => theme.colors.surface};
  border-radius: ${({ theme }) => theme.radii.lg};
  padding: ${({ theme }) => theme.spacing(2)};
  box-shadow: ${({ theme }) => theme.shadows.glow};
  display: grid;
  place-items: center;
  position: relative;
`;

const StaticImage = styled.img`
  width: 100%;
  height: auto;
  max-height: 70vh;
  object-fit: contain;
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid rgba(0, 0, 0, 0.08);
`;
const TimingCard = styled.div`
  border: 1px solid rgba(0, 0, 0, 0.05);
  background: ${({ theme }) => theme.colors.surfaceAlt};
  border-radius: ${({ theme }) => theme.radii.md};
  padding: ${({ theme }) => theme.spacing(1.5)};
  margin-bottom: ${({ theme }) => theme.spacing(2)};
  box-shadow: ${({ theme }) => theme.shadows.glow};
`;

const TimingHeader = styled.div`
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-weight: 700;
  color: ${({ theme }) => theme.colors.text};
  margin-bottom: ${({ theme }) => theme.spacing(1)};
`;

const TimingStatus = styled.span`
  color: #ff7a1a;
  font-weight: 700;
  font-size: 0.95rem;
`;

const TimingSummary = styled.p`
  margin: 0 0 ${({ theme }) => theme.spacing(1.5)} 0;
  color: ${({ theme }) => theme.colors.muted};
  font-size: 0.95rem;
`;

const TimingBar = styled.div`
  position: relative;
  height: 6px;
  border-radius: 999px;
  background: rgba(0, 0, 0, 0.08);
  overflow: visible;
  margin-top: ${({ theme }) => theme.spacing(1)};
  margin-bottom: ${({ theme }) => theme.spacing(2)};
`;

const TimingBand = styled.div<{ $left: number; $width: number }>`
  position: absolute;
  top: 0;
  height: 6px;
  border-radius: 999px;
  background: rgba(255, 122, 26, 0.22);
  left: ${({ $left }) => `${$left}%`};
  width: ${({ $width }) => `${$width}%`};
`;

const TimingTick = styled.div<{ $left: number; $color: string; $height?: number }>`
  position: absolute;
  bottom: -2px;
  left: ${({ $left }) => `${$left}%`};
  transform: translateX(-50%);
  width: 2px;
  height: ${({ $height }) => $height ?? 20}px;
  background: ${({ $color }) => $color};
  border-radius: 2px;
`;

const TimingLabel = styled.div<{ $left: number; $color: string; $placement?: 'top' | 'bottom' }>`
  position: absolute;
  ${({ $placement }) => ($placement === 'top' ? 'top: -32px;' : 'bottom: -32px;')}
  left: ${({ $left }) => `${$left}%`};
  transform: translateX(-50%);
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid ${({ $color }) => `${$color}55`};
  color: ${({ theme }) => theme.colors.text};
  border-radius: 10px;
  padding: 3px 8px;
  font-weight: 700;
  font-size: 0.78rem;
  box-shadow: ${({ theme }) => theme.shadows.glow};
  white-space: nowrap;
`;

const TimingLegend = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing(1)};
  margin-top: ${({ theme }) => theme.spacing(2.5)};
  flex-wrap: wrap;
`;

const TimingPill = styled.span<{ $color: string }>`
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 8px;
  border-radius: 8px;
  background: ${({ $color }) => `${$color}18`};
  color: ${({ theme }) => theme.colors.text};
  border: 1px solid ${({ $color }) => `${$color}4d`};
  font-weight: 600;
`;

interface Props {
  tabs: AnalysisTab[];
  videos: VideoSources;
  phases?: PhaseSegment[];
  frameCount?: number;
}

const defaultPhases: PhaseSegment[] = [
  { name: 'rest', startFrame: 0, endFrame: 25 },
  { name: 'draw', startFrame: 25, endFrame: 75 },
  { name: 'release', startFrame: 75, endFrame: 100 },
];

const phaseColors: Record<string, string> = {
  rest: '#b0b0b0',
  draw: '#ff7a1a',
  release: '#e53935',
};

const DRAW_LENGTH_GRACE_HW = 0.03;

type PhaseDisplaySegment = {
  label: string;
  width: number;
  color: string;
};

const buildPhaseSegments = (phases: PhaseSegment[], frameCount?: number): PhaseDisplaySegment[] => {
  const totalFrames =
    frameCount && frameCount > 0
      ? frameCount
      : phases.reduce((max, phase) => Math.max(max, phase.endFrame), 0) || 100;

  const segments = phases
    .map((phase) => {
      const start = Math.max(0, Math.min(phase.startFrame, totalFrames));
      const end = Math.max(start, Math.min(phase.endFrame, totalFrames));
      const width = totalFrames ? ((end - start) / totalFrames) * 100 : 0;
      const label = phase.name ?? 'phase';
      const color = phaseColors[label.toLowerCase()] ?? '#3c4a7a';
      return { label, width, color };
    })
    .filter((segment) => segment.width > 0);

  if (segments.length) {
    return segments;
  }

  // Fallback: evenly spaced if data is malformed.
  const safeTotal = Math.max(1, phases.length);
  return phases.map((phase) => ({
    label: phase.name,
    width: 100 / safeTotal,
    color: phaseColors[phase.name.toLowerCase()] ?? '#3c4a7a',
  }));
};

function clamp01(value: number): number {
  if (Number.isNaN(value) || !Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}

function GhostTimingMeter({ timing }: { timing: GhostTiming }) {
  const { drawSeconds, avgSeconds, stdSeconds, status, sampleCount, minSamples } = timing;
  const required = minSamples ?? MIN_DATASET_SAMPLES;
  if (avgSeconds === undefined || stdSeconds === undefined || stdSeconds <= 0 || drawSeconds === undefined) {
    return null;
  }

  const minVal = 0;
  const maxVal = Math.max(drawSeconds, avgSeconds + stdSeconds * 1.6, avgSeconds * 1.2, stdSeconds * 2, 0.5);
  const span = Math.max(0.001, maxVal - minVal);
  const bandStart = Math.max(minVal, avgSeconds - stdSeconds);
  const bandEnd = Math.min(maxVal, avgSeconds + stdSeconds);
  const toPct = (val: number) => clamp01((val - minVal) / span) * 100;

  const bandLeft = toPct(bandStart);
  const bandWidth = Math.max(2, toPct(bandEnd) - bandLeft);
  const avgLeft = toPct(avgSeconds);
  const drawLeft = toPct(drawSeconds);

  const statusText =
    status === 'too_short'
      ? 'Release was early relative to the dataset window.'
      : status === 'too_long'
        ? 'Release was late relative to the dataset window.'
        : sampleCount !== undefined && sampleCount < required
          ? 'Timing measured; dataset too small for judgment.'
          : 'Release timing is within the dataset window.';

  return (
    <TimingCard>
      <TimingHeader>
        <span>Ghost timing vs dataset</span>
        <TimingStatus>{statusText}</TimingStatus>
      </TimingHeader>
      <TimingSummary>
        Dataset window: {avgSeconds.toFixed(2)}s ± {stdSeconds.toFixed(2)}s · This shot: {drawSeconds.toFixed(2)}s
        {sampleCount !== undefined && sampleCount < required && (
          <span style={{ marginLeft: 8 }}>(Limited dataset: {sampleCount}/{required})</span>
        )}
      </TimingSummary>
      <TimingBar>
        <TimingBand $left={bandLeft} $width={bandWidth} />
        <TimingTick $left={avgLeft} $color="#ff7a1a" $height={18} />
        <TimingTick $left={drawLeft} $color="#2e1a0a" $height={24} />
        <TimingLabel $left={avgLeft} $color="#ff7a1a" $placement="top">
          Avg
        </TimingLabel>
        <TimingLabel $left={drawLeft} $color="#2e1a0a" $placement="bottom">
          {drawSeconds.toFixed(2)}s
        </TimingLabel>
      </TimingBar>
      <TimingLegend>
        <TimingPill $color="#ff7a1a">Dataset avg ± σ</TimingPill>
        <TimingPill $color="#2e1a0a">This shot</TimingPill>
      </TimingLegend>
    </TimingCard>
  );
}

type PhaseBannerProps = {
  title?: string;
  segments: PhaseDisplaySegment[];
  progress?: number | null;
  compact?: boolean;
};

function PhaseBanner({ title = 'Phase timeline', segments, progress, compact }: PhaseBannerProps) {
  const pointer = progress === undefined || progress === null ? undefined : clamp01(progress) * 100;
  return (
    <ProgressShell $compact={compact}>
      <ProgressHeader $compact={compact}>
        <span>{title}</span>
        {!compact && <span>{segments.map((segment) => segment.label).join(' · ')}</span>}
      </ProgressHeader>
      <ProgressBar>
        {segments.map((segment, idx) => (
          <ProgressSegment key={`${segment.label}-${idx}`} $color={segment.color} $width={segment.width} />
        ))}
        <ProgressPointer aria-hidden $position={pointer}>{'>'}</ProgressPointer>
      </ProgressBar>
    </ProgressShell>
  );
}

export function AnalysisWorkspace({ tabs, videos, phases = [], frameCount }: Props) {
  const [activeTab, setActiveTab] = useState(tabs[0]?.id ?? '');
  const [videoMode, setVideoMode] = useState<'original' | 'skeleton' | 'ghost'>('original');
  const [videoAspect, setVideoAspect] = useState(16 / 9);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(false);
  const [frameIndex, setFrameIndex] = useState(0);
  const [framePlayId, setFramePlayId] = useState(0);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const tabsRef = useRef<HTMLDivElement | null>(null);
  const updateScrollState = useCallback(() => {
    const el = tabsRef.current;
    if (!el) return;
    const { scrollLeft, scrollWidth, clientWidth } = el;
    const maxScroll = Math.max(0, scrollWidth - clientWidth);
    setCanScrollLeft(scrollLeft > 4);
    setCanScrollRight(scrollLeft < maxScroll - 4);
  }, []);
  const content = tabs.find((tab) => tab.id === activeTab);
  const phaseSegments = useMemo(
    () => buildPhaseSegments(phases.length ? phases : defaultPhases, frameCount),
    [phases, frameCount]
  );
  const videoOptions = useMemo(
    () => [
      { id: 'original', label: 'Original' },
      { id: 'skeleton', label: 'Skeleton' },
      { id: 'ghost', label: 'Ghost Overlay' },
    ] as const,
    []
  );

  useEffect(() => {
    const defaultTab = tabs.find((tab) => tab.id === 'ghost')?.id ?? tabs[0]?.id ?? '';
    setActiveTab(defaultTab);
  }, [tabs]);

  // Animate frames on tab activation for tabs that ship frame sequences.
  useEffect(() => {
    const frames = content?.frames;
    const hasFrames =
      activeTab === 'draw-force' || activeTab === 'draw-length' || activeTab === 'release' || activeTab === 'follow-through';
    if (!frames || frames.length === 0 || !hasFrames) {
      setFrameIndex(0);
      return;
    }
    setFrameIndex(0);
    let idx = 0;
    const interval = window.setInterval(() => {
      idx = Math.min(idx + 1, frames.length - 1);
      setFrameIndex(idx);
      if (idx >= frames.length - 1) {
        window.clearInterval(interval);
      }
    }, 220);
    return () => {
      window.clearInterval(interval);
    };
  }, [activeTab, content?.frames, framePlayId]);

  useEffect(() => {
    if (activeTab === 'ghost') {
      if (videos.original) {
        setVideoMode('original');
      } else if (videos.ghost) {
        setVideoMode('ghost');
      }
    } else if (activeTab === 'spine' && videos.skeleton) {
      setVideoMode('skeleton');
    } else if ((activeTab === 'draw-force' || activeTab === 'draw-length') && videos.original) {
      setVideoMode('original');
    }
  }, [activeTab, videos]);

  useEffect(() => {
    if (!videos[videoMode]) {
      const fallback = videoOptions.find((option) => videos[option.id]);
      if (fallback) {
        setVideoMode(fallback.id);
      }
    }
  }, [videoMode, videos, videoOptions]);

  useEffect(() => {
    const raf = requestAnimationFrame(updateScrollState);
    const timeout = window.setTimeout(updateScrollState, 50);
    const el = tabsRef.current;
    el?.addEventListener('scroll', updateScrollState);
    window.addEventListener('resize', updateScrollState);
    return () => {
      el?.removeEventListener('scroll', updateScrollState);
      window.removeEventListener('resize', updateScrollState);
      cancelAnimationFrame(raf);
      window.clearTimeout(timeout);
    };
  }, [tabs, updateScrollState]);

  const scrollTabs = (direction: 'left' | 'right') => {
    const el = tabsRef.current;
    if (!el) return;
    const delta = direction === 'left' ? -Math.max(120, el.clientWidth * 0.8) : Math.max(120, el.clientWidth * 0.8);
    el.scrollBy({ left: delta, behavior: 'smooth' });
    window.setTimeout(updateScrollState, 180);
  };

  const handleTabSelect = (tabId: string) => {
    setActiveTab(tabId);
    if (tabId === 'draw-force' || tabId === 'draw-length' || tabId === 'release' || tabId === 'follow-through') {
      setFramePlayId((prev) => prev + 1);
    }
  };

  const assetUrl = (source?: string | null) => {
    if (!source) return undefined;
    if (/^https?:\/\//i.test(source)) return source;
    if (source.startsWith('/')) return source;
    return `/assets/${source.replace(/^\.\//, '')}`;
  };

  const currentVideo = videos[videoMode];
  const displayImage = content?.frames && content.frames.length ? content.frames[Math.min(frameIndex, content.frames.length - 1)] : content?.image;
  const safeDuration = duration || 0;
  const phaseProgress = safeDuration ? clamp01(currentTime / safeDuration) : undefined;
  const isReportTab = activeTab === 'report';
  const isGhostTab = activeTab === 'ghost';
  const isSpineTab = activeTab === 'spine';
  const isDrawForceTab = activeTab === 'draw-force';
  const isDrawLengthTab = activeTab === 'draw-length';
  const isReleaseTab = activeTab === 'release';
  const isStabilityTab = activeTab === 'follow-through';

  useEffect(() => {
    const el = videoRef.current;
    if (el) {
      el.pause();
      el.currentTime = 0;
    }
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(0);
  }, [currentVideo]);

  const togglePlay = () => {
    const el = videoRef.current;
    if (!el) return;
    if (el.paused) {
      void el.play();
    } else {
      el.pause();
    }
  };

  const handleSeek = (value: number) => {
    const el = videoRef.current;
    if (!el || !duration) return;
    const next = Math.min(Math.max(value, 0), duration);
    el.currentTime = next;
    setCurrentTime(next);
  };

  const formatTime = (time: number) => {
    const clamped = Math.max(0, time);
    const minutes = Math.floor(clamped / 60)
      .toString()
      .padStart(2, '0');
    const seconds = Math.floor(clamped % 60)
      .toString()
      .padStart(2, '0');
    return `${minutes}:${seconds}`;
  };

  return (
    <Page>
      <HeaderBar>
        <BrandGroup>
          <LogoBadge src={logoBadge} alt="TPN logo" />
          <BrandText>STRAIGHT</BrandText>
        </BrandGroup>
        <NavActions aria-label="Primary navigation">
          <IconButton href="/" aria-label="Home">
            <IconSvg viewBox="0 0 24 24" role="img" aria-hidden="true">
              <path d="M12 3.172 3.5 10.5V21h6v-5h5v5h6v-10.5L12 3.172z" />
            </IconSvg>
          </IconButton>
          <IconButton href="https://github.com/mont1y/STRAIGHT" target="_blank" rel="noreferrer" aria-label="GitHub repository">
            <IconSvg viewBox="0 0 24 24" role="img" aria-hidden="true">
              <path d="M12 .5C5.65.5.5 5.65.5 12c0 5.09 3.29 9.4 7.86 10.93.58.1.79-.25.79-.56 0-.28-.01-1.02-.02-2-3.2.7-3.88-1.54-3.88-1.54-.53-1.35-1.28-1.7-1.28-1.7-1.05-.72.08-.7.08-.7 1.16.08 1.78 1.2 1.78 1.2 1.03 1.77 2.7 1.26 3.35.96.1-.75.4-1.26.72-1.55-2.55-.29-5.23-1.28-5.23-5.72 0-1.26.45-2.29 1.2-3.1-.12-.29-.52-1.45.12-3.02 0 0 .98-.31 3.2 1.18.93-.26 1.93-.4 2.92-.4.99 0 1.99.14 2.92.4 2.22-1.49 3.2-1.18 3.2-1.18.64 1.57.24 2.73.12 3.02.75.81 1.2 1.84 1.2 3.1 0 4.45-2.69 5.43-5.25 5.71.41.36.77 1.08.77 2.18 0 1.58-.02 2.86-.02 3.25 0 .31.21.67.8.55A10.52 10.52 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5Z" />
            </IconSvg>
          </IconButton>
        </NavActions>
      </HeaderBar>
      <Shell>
        <VideoPanel>
          <TabBar>
            <TabsArrow
              type="button"
              $side="left"
              aria-label="Scroll tabs left"
              disabled={!canScrollLeft}
              onClick={() => scrollTabs('left')}
            >
              ‹
            </TabsArrow>
            <TabScrollArea ref={tabsRef}>
              <Tabs role="tablist">
                {tabs.map((tab) => (
                  <TabButton
                    key={tab.id}
                    role="tab"
                    aria-selected={tab.id === activeTab}
                    $active={tab.id === activeTab}
                    onClick={() => handleTabSelect(tab.id)}
                  >
                    {tab.label}
                  </TabButton>
                ))}
              </Tabs>
            </TabScrollArea>
            <TabsArrow
              type="button"
              $side="right"
              aria-label="Scroll tabs right"
              disabled={!canScrollRight}
              onClick={() => scrollTabs('right')}
            >
              ›
            </TabsArrow>
          </TabBar>
          {(isSpineTab || isDrawForceTab || isDrawLengthTab || isReleaseTab || isStabilityTab) && displayImage ? (
            <StaticImageShell>
              <StaticImage src={assetUrl(displayImage)} alt={`${content?.label ?? 'analysis'} visualization`} />
            </StaticImageShell>
          ) : (
            <>
              <VideoSurface $aspect={videoAspect}>
                {currentVideo ? (
                  <>
                    <VideoElement
                      key={currentVideo}
                      ref={videoRef}
                      src={assetUrl(currentVideo)}
                      controls={false}
                      playsInline
                      preload="metadata"
                      aria-label={`${videoMode} video`}
                      onLoadedMetadata={(event) => {
                        const { videoWidth, videoHeight, duration: dur } = event.currentTarget;
                        if (videoWidth && videoHeight) {
                          setVideoAspect(videoWidth / videoHeight);
                        }
                        if (!Number.isNaN(dur)) {
                          setDuration(dur);
                        }
                      }}
                      onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
                      onPlay={() => setIsPlaying(true)}
                      onPause={() => setIsPlaying(false)}
                    />
                    <Controls>
                      <ControlButton type="button" onClick={togglePlay} disabled={!safeDuration}>
                        {isPlaying ? 'Pause' : 'Play'}
                      </ControlButton>
                      <Seek
                        type="range"
                        min={0}
                        max={safeDuration || 0}
                        step="0.01"
                        value={currentTime}
                        onChange={(e) => handleSeek(Number(e.target.value))}
                        disabled={!safeDuration}
                      />
                      <Timecode>
                        {formatTime(currentTime)} / {formatTime(safeDuration)}
                      </Timecode>
                    </Controls>
                  </>
                ) : (
                  <VideoFallback>No video available for {videoMode}.</VideoFallback>
                )}
              </VideoSurface>
              <ToggleGroup role="group" aria-label="Video view selector">
                {videoOptions.map((option) => (
                  <ToggleButton
                    key={option.id}
                    type="button"
                    disabled={!videos[option.id]}
                    $active={videoMode === option.id}
                    onClick={() => setVideoMode(option.id)}
                  >
                    {option.label}
                  </ToggleButton>
                ))}
              </ToggleGroup>
              <PhaseBanner segments={phaseSegments} progress={phaseProgress} />
            </>
          )}
      </VideoPanel>
        <MarkdownPanel>
          {isSpineTab && content?.spineSummary && (
            <SpineCard>
              <SpineMetric>
                <SpineLabel>Normalized distance</SpineLabel>
                <SpineValue>
                {content.spineSummary.normalizedDistance !== undefined
                  ? content.spineSummary.normalizedDistance.toFixed(2)
                  : 'N/A'}
              </SpineValue>
            </SpineMetric>
              {content.spineSummary.status && <SpineChip>{content.spineSummary.status}</SpineChip>}
            </SpineCard>
          )}
          {isDrawForceTab && content?.drawForceSummary && (
            <DrawForceCard>
              <StatRow>
                <SpineMetric>
                  <SpineLabel>Angle</SpineLabel>
                  <SpineValue>
                    {content.drawForceSummary.angleDeg !== undefined
                      ? `${content.drawForceSummary.angleDeg.toFixed(2)}°`
                      : 'N/A'}
                  </SpineValue>
                  {(content.drawForceSummary.angleAvg !== undefined || content.drawForceSummary.angleStd !== undefined) && (
                    <SpineLabel>
                      Avg {content.drawForceSummary.angleAvg?.toFixed(2) ?? 'N/A'}° · σ{' '}
                      {content.drawForceSummary.angleStd?.toFixed(2) ?? 'N/A'}°
                    </SpineLabel>
                  )}
                  {content.drawForceSummary.angleSampleCount !== undefined &&
                    content.drawForceSummary.minSamplesRequired !== undefined &&
                    content.drawForceSummary.angleSampleCount < content.drawForceSummary.minSamplesRequired && (
                      <SpineLabel>
                        Dataset shots: {content.drawForceSummary.angleSampleCount}/{content.drawForceSummary.minSamplesRequired}
                      </SpineLabel>
                  )}
                </SpineMetric>
              </StatRow>
              {content.drawForceSummary.status && (
                <SpineChip $tone={content.drawForceSummary.status !== 'Good' ? 'warn' : undefined}>
                  {content.drawForceSummary.status}
                </SpineChip>
              )}
            </DrawForceCard>
          )}
          {isDrawLengthTab && content?.drawForceSummary && (
            <DrawForceCard>
              <StatRow>
                <SpineMetric>
                  <SpineLabel>Draw length</SpineLabel>
                  <SpineValue>
                    {content.drawForceSummary.drawLengthHw !== undefined
                      ? `${content.drawForceSummary.drawLengthHw.toFixed(2)} hw`
                      : 'N/A'}
                  </SpineValue>
                  {(content.drawForceSummary.drawLengthAvg !== undefined ||
                    content.drawForceSummary.drawLengthStd !== undefined) && (
                    <SpineLabel>
                      Avg {content.drawForceSummary.drawLengthAvg?.toFixed(2) ?? 'N/A'} · σ{' '}
                      {content.drawForceSummary.drawLengthStd?.toFixed(2) ?? 'N/A'} + grace {DRAW_LENGTH_GRACE_HW.toFixed(2)}
                    </SpineLabel>
                  )}
                  {content.drawForceSummary.drawLengthSampleCount !== undefined &&
                    content.drawForceSummary.minSamplesRequired !== undefined &&
                    content.drawForceSummary.drawLengthSampleCount < content.drawForceSummary.minSamplesRequired && (
                      <SpineLabel>
                        Dataset shots: {content.drawForceSummary.drawLengthSampleCount}/
                        {content.drawForceSummary.minSamplesRequired}
                      </SpineLabel>
                  )}
                </SpineMetric>
              </StatRow>
              {content.drawForceSummary.status && (
                <SpineChip $tone={content.drawForceSummary.status !== 'Good' ? 'warn' : undefined}>
                  {content.drawForceSummary.status}
                </SpineChip>
              )}
            </DrawForceCard>
          )}
          {isReleaseTab && content?.postReleaseDrawSummary && (
            <DrawForceCard>
              <StatRow>
                <SpineMetric>
                  <SpineLabel>Pre nose→wrist</SpineLabel>
                  <SpineValue>
                    {content.postReleaseDrawSummary.prePx !== undefined
                      ? `${content.postReleaseDrawSummary.prePx.toFixed(1)} px`
                      : 'N/A'}
                  </SpineValue>
                </SpineMetric>
                <SpineMetric>
                  <SpineLabel>Post nose→wrist</SpineLabel>
                  <SpineValue>
                    {content.postReleaseDrawSummary.postPx !== undefined
                      ? `${content.postReleaseDrawSummary.postPx.toFixed(1)} px`
                      : 'N/A'}
                  </SpineValue>
                </SpineMetric>
              </StatRow>
              <StatRow>
                <SpineMetric>
                  <SpineLabel>Change</SpineLabel>
                  <SpineValue>
                    {content.postReleaseDrawSummary.changePct !== undefined
                      ? `${content.postReleaseDrawSummary.changePct.toFixed(1)}%`
                      : 'N/A'}
                  </SpineValue>
                  {content.postReleaseDrawSummary.thresholdPct !== undefined && (
                    <SpineLabel>Target &gt; {content.postReleaseDrawSummary.thresholdPct.toFixed(1)}%</SpineLabel>
                  )}
                </SpineMetric>
              </StatRow>
              {content.postReleaseDrawSummary.status && (
                <SpineChip $tone={content.postReleaseDrawSummary.status !== 'Good' ? 'warn' : undefined}>
                  {content.postReleaseDrawSummary.status}
                </SpineChip>
              )}
            </DrawForceCard>
          )}
          {isStabilityTab && content?.postReleaseBowSummary && (
            <DrawForceCard>
              <StatRow>
                <SpineMetric>
                  <SpineLabel>Bow arm pre</SpineLabel>
                  <SpineValue>
                    {content.postReleaseBowSummary.preDeg !== undefined
                      ? `${content.postReleaseBowSummary.preDeg.toFixed(1)}°`
                      : 'N/A'}
                  </SpineValue>
                </SpineMetric>
                <SpineMetric>
                  <SpineLabel>Bow arm post</SpineLabel>
                  <SpineValue>
                    {content.postReleaseBowSummary.postDeg !== undefined
                      ? `${content.postReleaseBowSummary.postDeg.toFixed(1)}°`
                      : 'N/A'}
                  </SpineValue>
                </SpineMetric>
              </StatRow>
              <StatRow>
                <SpineMetric>
                  <SpineLabel>Delta</SpineLabel>
                  <SpineValue>
                    {content.postReleaseBowSummary.deltaDeg !== undefined
                      ? `${content.postReleaseBowSummary.deltaDeg.toFixed(1)}°`
                      : 'N/A'}
                  </SpineValue>
                  {content.postReleaseBowSummary.thresholdDeg !== undefined && (
                    <SpineLabel>Target &lt; {content.postReleaseBowSummary.thresholdDeg.toFixed(1)}°</SpineLabel>
                  )}
                </SpineMetric>
              </StatRow>
              {content.postReleaseBowSummary.status && (
                <SpineChip $tone={content.postReleaseBowSummary.status !== 'Good' ? 'warn' : undefined}>
                  {content.postReleaseBowSummary.status}
                </SpineChip>
              )}
            </DrawForceCard>
          )}
          {isGhostTab && content?.ghostTiming && <GhostTimingMeter timing={content.ghostTiming} />}
          {isReportTab && <PhaseBanner title="Phase banner" segments={phaseSegments} compact />}
          {content ? <ReactMarkdown>{content.markdown}</ReactMarkdown> : <p>Select an analysis tab to view details.</p>}
        </MarkdownPanel>
      </Shell>
    </Page>
  );
}
