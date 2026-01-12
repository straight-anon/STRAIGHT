import { useEffect, useRef, useState, type ReactNode } from 'react';
import styled, { ThemeProvider } from 'styled-components';
import { GlobalStyle } from './styles/global';
import { theme } from './theme';
import { ProfileOnboarding } from './components/ProfileOnboarding';
import { DashboardHome } from './components/DashboardHome';
import { PipelineLoader } from './components/PipelineLoader';
import { ModeSelect } from './components/ModeSelect';
import { CoachChat } from './components/CoachChat';
import {
  AnalysisWorkspace,
  type AnalysisTab,
  type VideoSources,
  type GhostTiming,
  type SpineSummary,
  type DrawForceSummary,
  type PostReleaseBowSummary,
  type PostReleaseDrawSummary,
} from './components/AnalysisWorkspace';
import { GhostReferenceSelector } from './components/GhostReferenceSelector';
import logoIntro from './assets/logo_combined.gif';
import logoIntroStill from './assets/lastFrame.png';
import {
  defaultPipelineSteps,
  fetchAnalysisResults,
  getProfile,
  pollPipelineStatus,
  regenerateReports,
  type RegenerateReportsResult,
  saveProfile,
  uploadTrainingVideo,
  type AnalysisResult,
  type PipelineProgress,
  type PhaseSegment,
  type Profile,
} from './services/api';

const Centered = styled.div`
  min-height: 100vh;
  display: grid;
  place-content: center;
  font-family: ${({ theme }) => theme.fonts.heading};
`;

type ViewState =
  | 'loading'
  | 'onboarding'
  | 'dashboard'
  | 'modeSelect'
  | 'processing'
  | 'complete'
  | 'results'
  | 'ghost-selector'
  | 'coach-chat';

const INTRO_PLAY_MS = 5600; // total GIF length (~5.34s) plus a small buffer
const INTRO_FADE_MS = 1400;
const MIN_DATASET_SAMPLES = 5;
const POST_RELEASE_DRAW_THRESHOLD = 45;

function App() {
  const [introSeen] = useState(false);
  const [showIntro, setShowIntro] = useState(true);
  const [renderIntro, setRenderIntro] = useState(true);
  const [showGif, setShowGif] = useState(true);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [view, setView] = useState<ViewState>('loading');
  const [pipelineSteps, setPipelineSteps] = useState<PipelineProgress[]>(defaultPipelineSteps());
  const [analysisTabs, setAnalysisTabs] = useState<AnalysisTab[]>([]);
  const [videoSources, setVideoSources] = useState<VideoSources>({});
  const [phases, setPhases] = useState<PhaseSegment[]>([]);
  const [frameCount, setFrameCount] = useState<number | undefined>(undefined);
  const [pipelineError, setPipelineError] = useState<string | null>(null);
  const [lastRunId, setLastRunId] = useState<string | null>(null);
  const [pendingUpload, setPendingUpload] = useState<File | null>(null);
  const pollingRef = useRef<number | null>(null);
  const introFreezeRef = useRef<number | null>(null);
  const introRemoveRef = useRef<number | null>(null);

  const stripImages = (text?: string | null) => {
    if (!text) return '';
    return text.replace(/!\[[^\]]*]\([^)]+\)/g, '').trim();
  };

  const toNumber = (val: string | number | undefined): number | undefined => {
    if (typeof val === 'number' && Number.isFinite(val)) return val;
    if (typeof val === 'string') {
      const match = val.match(/-?\d+(\.\d+)?/);
      if (match) {
        const num = Number(match[0]);
        if (Number.isFinite(num)) return num;
      }
    }
    return undefined;
  };

  const hasDatasetWindow = (
    sampleCount: number | undefined,
    avg?: number,
    std?: number,
    minSamples = MIN_DATASET_SAMPLES,
  ) => {
    if (avg === undefined || std === undefined || std <= 0) return false;
    if (sampleCount === undefined || sampleCount === null) return true;
    return sampleCount >= minSamples;
  };

  const parseGhostTiming = (markdown?: string): { cleaned: string; timing?: GhostTiming } => {
    if (!markdown) {
      return { cleaned: '', timing: undefined };
    }
    const lines = markdown.split('\n');
    const startIdx = lines.findIndex((line) => line.trim() === 'DATA_START');
    const endIdx = lines.findIndex((line, idx) => idx > startIdx && line.trim() === 'DATA_END');
    if (startIdx === -1 || endIdx === -1 || endIdx <= startIdx) {
      return { cleaned: markdown, timing: undefined };
    }

    const dataLines = lines.slice(startIdx + 1, endIdx);
    const cleanedLines = [...lines.slice(0, startIdx), ...lines.slice(endIdx + 1)];
    const kv: Record<string, number | string> = {};
    dataLines.forEach((line) => {
      const match = line.match(/^\s*([^=]+)=(.+)$/);
      if (!match) return;
      const [, key, rawVal] = match;
      const maybeNum = toNumber(rawVal.trim());
      kv[key.trim()] = maybeNum !== undefined ? maybeNum : rawVal.trim();
    });

    const timing: GhostTiming = {
      drawSeconds: typeof kv.draw_to_release_seconds === 'number' ? kv.draw_to_release_seconds : undefined,
      avgSeconds:
        typeof kv.dataset_draw_to_release_avg_seconds === 'number' ? kv.dataset_draw_to_release_avg_seconds : undefined,
      stdSeconds:
        typeof kv.dataset_draw_to_release_std_seconds === 'number' ? kv.dataset_draw_to_release_std_seconds : undefined,
      status: typeof kv.timing_status === 'string' ? kv.timing_status : undefined,
      sampleCount:
        typeof kv.dataset_draw_to_release_sample_count === 'number'
          ? Number(kv.dataset_draw_to_release_sample_count)
          : undefined,
      minSamples:
        typeof kv.dataset_min_samples_required === 'number'
          ? Number(kv.dataset_min_samples_required)
          : MIN_DATASET_SAMPLES,
    };

    const cleaned = cleanedLines.join('\n').replace(/\n{3,}/g, '\n\n').trim();
    return { cleaned, timing };
  };

  const parseDataBlock = (markdown?: string | null): { cleaned: string; data: Record<string, number | string> } => {
    if (!markdown) return { cleaned: '', data: {} };
    const lines = markdown.split('\n');
    const startIdx = lines.findIndex((line) => line.trim() === 'DATA_START');
    const endIdx = lines.findIndex((line, idx) => idx > startIdx && line.trim() === 'DATA_END');
    if (startIdx === -1 || endIdx === -1 || endIdx <= startIdx) {
      return { cleaned: markdown.trim(), data: {} };
    }
    const dataLines = lines.slice(startIdx + 1, endIdx);
    const cleanedLines = [...lines.slice(0, startIdx), ...lines.slice(endIdx + 1)];
    const kv: Record<string, number | string> = {};
    dataLines.forEach((line) => {
      const match = line.match(/^\s*([^=]+)=(.+)$/);
      if (!match) return;
      const [, key, rawVal] = match;
      const maybeNum = toNumber(rawVal.trim());
      kv[key.trim()] = maybeNum !== undefined ? maybeNum : rawVal.trim();
    });
    const cleaned = cleanedLines.join('\n').replace(/\n{3,}/g, '\n\n').trim();
    return { cleaned, data: kv };
  };

  const statusFromWindow = (value?: number, avg?: number, std?: number, grace = 0): string | undefined => {
    if (value === undefined || avg === undefined || std === undefined || std <= 0) return undefined;
    const diff = Math.abs(value - avg);
    return diff <= std + grace ? 'Good' : 'Needs attention';
  };

  const parseSpineSummary = (markdown?: string | null): SpineSummary | undefined => {
    if (!markdown) return undefined;
    const normMatch = markdown.match(/normalized distance\s+([0-9.]+)/i);
    const normalizedDistance = normMatch ? Number(normMatch[1]) : undefined;
    let status: string | undefined;
    if (/good|straight/i.test(markdown)) status = 'Good';
    if (/bad|curved|lean/i.test(markdown)) status = status ?? 'Needs improvement';
    return normalizedDistance !== undefined || status ? { normalizedDistance, status } : undefined;
  };

  const parseDrawForceSummary = (markdown?: string | null): DrawForceSummary | undefined => {
    if (!markdown) return undefined;
    const { data } = parseDataBlock(markdown);
    const angleDeg = typeof data.draw_force_angle_deg === 'number' ? data.draw_force_angle_deg : undefined;
    const angleAvg = typeof data.draw_force_angle_avg_deg === 'number' ? data.draw_force_angle_avg_deg : undefined;
    const angleStd = typeof data.draw_force_angle_std_deg === 'number' ? data.draw_force_angle_std_deg : undefined;
    const angleSampleCount =
      typeof data.draw_force_angle_sample_count === 'number' ? Number(data.draw_force_angle_sample_count) : undefined;
    const minSamples =
      typeof data.draw_force_min_samples_required === 'number'
        ? Number(data.draw_force_min_samples_required)
        : MIN_DATASET_SAMPLES;
    const lengthMatch = markdown.match(/draw length:\s*([0-9.]+)/i);
    const drawLengthHw = lengthMatch ? Number(lengthMatch[1]) : undefined;
    const hasAngleWindow = hasDatasetWindow(angleSampleCount, angleAvg, angleStd, minSamples);
    const normalizedAngleAvg = hasAngleWindow ? angleAvg : undefined;
    const normalizedAngleStd = hasAngleWindow ? angleStd : undefined;
    let status = hasAngleWindow ? statusFromWindow(angleDeg, normalizedAngleAvg, normalizedAngleStd) : 'Not enough data';
    if (!status && hasAngleWindow) {
      if (/good|aligned|solid/i.test(markdown)) status = 'Good';
      if (/warning|issue|poor/i.test(markdown)) status = status ?? 'Needs attention';
    }
    return angleDeg !== undefined || drawLengthHw !== undefined || status
      ? {
          angleDeg,
          angleAvg: normalizedAngleAvg,
          angleStd: normalizedAngleStd,
          angleSampleCount,
          minSamplesRequired: minSamples,
          drawLengthHw,
          status,
        }
      : undefined;
  };

  const parseDrawLengthSummary = (markdown?: string | null): DrawForceSummary | undefined => {
    if (!markdown) return undefined;
    const { data } = parseDataBlock(markdown);
    const drawLengthHw =
      typeof data.draw_length_hipwidths === 'number' ? data.draw_length_hipwidths : undefined;
    const drawLengthAvg =
      typeof data.draw_length_avg_hipwidths === 'number' ? data.draw_length_avg_hipwidths : undefined;
    const drawLengthStd =
      typeof data.draw_length_std_hipwidths === 'number' ? data.draw_length_std_hipwidths : undefined;
    const drawLengthSampleCount =
      typeof data.draw_length_sample_count === 'number' ? Number(data.draw_length_sample_count) : undefined;
    const minSamples =
      typeof data.draw_length_min_samples_required === 'number'
        ? Number(data.draw_length_min_samples_required)
        : MIN_DATASET_SAMPLES;
    const hasLengthWindow = hasDatasetWindow(drawLengthSampleCount, drawLengthAvg, drawLengthStd, minSamples);
    const normalizedAvg = hasLengthWindow ? drawLengthAvg : undefined;
    const normalizedStd = hasLengthWindow ? drawLengthStd : undefined;
    let status = hasLengthWindow ? statusFromWindow(drawLengthHw, normalizedAvg, normalizedStd, 0.1) : 'Not enough data';
    if (!status && hasLengthWindow) {
      if (/good|within/i.test(markdown)) status = 'Good';
      if (/short|long|issue|poor/i.test(markdown)) status = status ?? 'Needs attention';
    }
    return drawLengthHw !== undefined || status
      ? {
          drawLengthHw,
          drawLengthAvg: normalizedAvg,
          drawLengthStd: normalizedStd,
          drawLengthSampleCount,
          minSamplesRequired: minSamples,
          status,
        }
      : undefined;
  };

  const parsePostReleaseDrawSummary = (markdown?: string | null): PostReleaseDrawSummary | undefined => {
    if (!markdown) return undefined;
    const { data } = parseDataBlock(markdown);
    const prePx = toNumber(data.nose_draw_length_pre_px as string | number | undefined);
    const postPx = toNumber(data.nose_draw_length_follow_px as string | number | undefined);
    const changePct = toNumber(data.nose_draw_length_change_pct as string | number | undefined);
    const thresholdPct =
      toNumber(data.nose_draw_length_good_threshold_pct as string | number | undefined) ??
      POST_RELEASE_DRAW_THRESHOLD;
    let status: string | undefined;
    if (changePct !== undefined && thresholdPct !== undefined) {
      status = changePct > thresholdPct ? 'Good' : 'Needs attention';
    }
    return prePx !== undefined || postPx !== undefined || changePct !== undefined
      ? { prePx, postPx, changePct, thresholdPct, status }
      : undefined;
  };

  const parsePostReleaseBowSummary = (markdown?: string | null): PostReleaseBowSummary | undefined => {
    if (!markdown) return undefined;
    const { data } = parseDataBlock(markdown);
    const preDeg = toNumber(data.bow_torso_angle_pre_deg as string | number | undefined);
    const postDeg = toNumber(data.bow_torso_angle_follow_deg as string | number | undefined);
    const deltaDeg = toNumber(data.bow_torso_angle_delta_deg as string | number | undefined);
    const thresholdDeg = toNumber(data.bow_torso_angle_threshold_deg as string | number | undefined);
    let status: string | undefined;
    if (deltaDeg !== undefined && thresholdDeg !== undefined) {
      status = deltaDeg < thresholdDeg ? 'Good' : 'Needs attention';
    }
    return preDeg !== undefined || postDeg !== undefined || deltaDeg !== undefined
      ? { preDeg, postDeg, deltaDeg, thresholdDeg, status }
      : undefined;
  };

  useEffect(() => {
    introFreezeRef.current = window.setTimeout(() => {
      setShowGif(false);
      setShowIntro(false);
    }, INTRO_PLAY_MS);
    introRemoveRef.current = window.setTimeout(() => setRenderIntro(false), INTRO_PLAY_MS + INTRO_FADE_MS + 120);
    (async () => {
      const existing = await getProfile();
      if (existing) {
        setProfile(existing);
        setView('dashboard');
      } else {
        setView('onboarding');
      }
    })();
    return () => {
      if (pollingRef.current) {
        window.clearTimeout(pollingRef.current);
      }
      if (introFreezeRef.current) {
        window.clearTimeout(introFreezeRef.current);
        introFreezeRef.current = null;
      }
      if (introRemoveRef.current) {
        window.clearTimeout(introRemoveRef.current);
        introRemoveRef.current = null;
      }
    };
  }, [introSeen]);

  const skipIntro = () => {
    if (!renderIntro) return;
    if (introFreezeRef.current) {
      window.clearTimeout(introFreezeRef.current);
      introFreezeRef.current = null;
    }
    if (introRemoveRef.current) {
      window.clearTimeout(introRemoveRef.current);
      introRemoveRef.current = null;
    }
    setShowGif(false);
    setShowIntro(false);
    introRemoveRef.current = window.setTimeout(() => setRenderIntro(false), INTRO_FADE_MS + 80);
  };

  const handleProfileComplete = async (payload: Profile) => {
    const saved = await saveProfile(payload);
    setProfile(saved);
    setView('dashboard');
  };

  const buildTabs = (result: AnalysisResult): AnalysisTab[] => {
    const safeParse = <T,>(fn: () => T): T | undefined => {
      try {
        return fn();
      } catch (err) {
        console.error('Failed to parse analysis content', err);
        return undefined;
      }
    };
    const tabs: AnalysisTab[] = [];
    const { markdown, assets } = result;
    const toFrameList = (frames?: string[]) => {
      return Array.isArray(frames) ? frames.filter(Boolean) : [];
    };
    const drawForceFrames = toFrameList(assets?.drawForceFrames);
    const drawLengthFrames = toFrameList(assets?.drawLengthFrames);
    const releaseFrames = toFrameList(assets?.releaseFrames);
    const followThroughFrames = toFrameList(assets?.followThroughFrames);
    // Ghost overlay
    if (markdown.ghost) {
      const { cleaned, timing } = parseGhostTiming(markdown.ghost);
      tabs.push({ id: 'ghost', label: 'Ghost Overlay', markdown: cleaned, ghostTiming: timing });
    }
    // Spine straightness
    tabs.push({
      id: 'spine',
      label: 'Spine Straightness',
      markdown: stripImages(markdown.spine) || 'Spine analysis unavailable.',
      image: assets.spineImage,
      spineSummary: parseSpineSummary(markdown.spine),
    });
    // Draw-force line
    tabs.push({
      id: 'draw-force',
      label: 'Draw-Force Line',
      markdown: stripImages(parseDataBlock(markdown.drawForce).cleaned) || 'Draw-force analysis unavailable.',
      image: assets.drawForceImage || assets.original,
      frames: drawForceFrames,
      drawForceSummary: safeParse(() => parseDrawForceSummary(markdown.drawForce)),
    });
    // Draw length
    tabs.push({
      id: 'draw-length',
      label: 'Draw Length',
      markdown: stripImages(parseDataBlock(markdown.drawLength).cleaned) || 'Draw length analysis unavailable.',
      image: assets.drawLengthImage || assets.drawForceImage || assets.original,
      frames: drawLengthFrames,
      drawForceSummary: safeParse(() => parseDrawLengthSummary(markdown.drawLength)),
    });
    // Follow-through
    const releaseMd = markdown.release ?? markdown.postRelease;
    const followThroughMd = markdown.followThrough ?? markdown.postRelease;
    const releaseParsed = parseDataBlock(releaseMd);
    const followParsed = parseDataBlock(followThroughMd);
    tabs.push({
      id: 'release',
      label: 'Release',
      markdown: stripImages(releaseParsed.cleaned) || 'Release analysis unavailable.',
      image: assets.followThroughImage || assets.postReleaseImage,
      frames: followThroughFrames,
      postReleaseDrawSummary: safeParse(() => parsePostReleaseDrawSummary(releaseMd)),
    });
    tabs.push({
      id: 'follow-through',
      label: 'Follow-Through Stability',
      markdown: stripImages(followParsed.cleaned) || 'Follow-through analysis unavailable.',
      image: assets.releaseImage || assets.postReleaseImage,
      frames: releaseFrames,
      postReleaseBowSummary: safeParse(() => parsePostReleaseBowSummary(followThroughMd)),
    });
    return tabs;
  };

  const beginUpload = async (useShotTrainer: boolean) => {
    if (!pendingUpload) {
      setPipelineError('No video selected. Please upload again.');
      setView(profile ? 'dashboard' : 'onboarding');
      return;
    }
    setPipelineError(null);
    setAnalysisTabs([]);
    setVideoSources({});
    setPhases([]);
    setFrameCount(undefined);
    setPipelineSteps(
      defaultPipelineSteps().map((step) => (step.id === 'upload' ? { ...step, status: 'active' as const } : step)),
    );
    setView('processing');
    const { uploadId } = await uploadTrainingVideo(pendingUpload, useShotTrainer);
    setPendingUpload(null);
    setLastRunId(uploadId);
    setPipelineSteps((prev) => prev.map((step) => (step.id === 'upload' ? { ...step, status: 'done' as const } : step)));

    const poll = async () => {
      try {
        const status = await pollPipelineStatus(uploadId);
        setPipelineSteps(status.steps);
        if (status.error) {
          setPipelineError(status.error);
          setView('dashboard');
          return;
        }
        if (!status.done) {
          pollingRef.current = window.setTimeout(poll, 1500);
          return;
        }
        if (pollingRef.current) {
          window.clearTimeout(pollingRef.current);
        }
        const analysis = await fetchAnalysisResults(uploadId);
        setAnalysisTabs(buildTabs(analysis));
        setVideoSources({
          original: analysis.assets.original,
          skeleton: analysis.assets.skeleton,
          ghost: analysis.assets.ghost,
        });
        setPhases(analysis.phases ?? []);
        setFrameCount(analysis.frameCount);
        setView('complete');
      } catch (err) {
        console.error(err);
        setPipelineError('Pipeline failed. Please check the backend logs.');
        setView('dashboard');
      }
    };

    poll();
  };

  const handleUpload = (file: File) => {
    setPendingUpload(file);
    setView('modeSelect');
  };

  const handleViewAnalysis = async (analysisId: string) => {
    try {
      setPipelineError(null);
      const analysis = await fetchAnalysisResults(analysisId);
      setAnalysisTabs(buildTabs(analysis));
      setVideoSources({
        original: analysis.assets.original,
        skeleton: analysis.assets.skeleton,
        ghost: analysis.assets.ghost,
      });
      setPhases(analysis.phases ?? []);
      setFrameCount(analysis.frameCount);
      setView('results');
    } catch (err) {
      console.error(err);
      setPipelineError('Unable to load analysis from assets.');
      setView('dashboard');
    }
  };

  const handleRegenerateReports = async (analysisId: string): Promise<RegenerateReportsResult> => {
    try {
      return await regenerateReports(analysisId);
    } catch (err) {
      console.error(err);
      setPipelineError('Unable to regenerate reports for this run.');
      throw err;
    }
  };

  const openGhostSelector = () => setView('ghost-selector');
  const openCoachChat = () => {
    setView('coach-chat');
  };
  const closeCoachChat = () => setView('dashboard');

  let screen: ReactNode;
  if (view === 'loading') {
    screen = <Centered>Loading STRAIGHT…</Centered>;
  } else if (view === 'onboarding') {
    screen = <ProfileOnboarding onComplete={handleProfileComplete} />;
  } else if (view === 'dashboard' && profile) {
    screen = (
      <DashboardHome
        profileName={profile.name}
        onUpload={handleUpload}
        onViewAnalysis={handleViewAnalysis}
        onRegenerateReports={handleRegenerateReports}
        onOpenCoachChat={openCoachChat}
        onOpenGhostSelector={openGhostSelector}
      />
    );
  } else if (view === 'coach-chat') {
    screen = <CoachChat profileName={profile?.name} onBack={closeCoachChat} />;
  } else if (view === 'modeSelect') {
    screen = (
      <ModeSelect
        filename={pendingUpload?.name}
        onSelect={(useShotTrainer) => beginUpload(useShotTrainer)}
        onCancel={() => {
          setPendingUpload(null);
          setView(profile ? 'dashboard' : 'onboarding');
        }}
      />
    );
  } else if (view === 'processing') {
    screen = (
      <PipelineLoader
        steps={pipelineSteps}
        tipText={pipelineError ?? 'Your video is traveling through the STRAIGHT pipeline. Sit tight while we fetch your metrics.'}
      />
    );
  } else if (view === 'complete') {
    screen = (
      <CompletionCard>
        <CompletionPanel>
          <CompletionBadge>{lastRunId ? `Run #${lastRunId.slice(0, 6)}` : 'Latest run'}</CompletionBadge>
          <CompletionTitle>Report ready to review</CompletionTitle>
          <CompletionSubtitle>
            We exported your visuals and markdown to the dashboard. Jump back to browse the full report.
          </CompletionSubtitle>
          <CompletionActions>
            <CompletionButton type="button" onClick={() => setView(profile ? 'dashboard' : 'onboarding')}>
              Go to dashboard
            </CompletionButton>
          </CompletionActions>
        </CompletionPanel>
      </CompletionCard>
    );
  } else if (view === 'results') {
    screen = (
      <AnalysisWorkspace
        tabs={analysisTabs}
        videos={videoSources}
        phases={phases}
        frameCount={frameCount}
      />
    );
  } else if (view === 'ghost-selector') {
    screen = <GhostReferenceSelector onBack={() => setView(profile ? 'dashboard' : 'onboarding')} />;
  } else {
    screen = <Centered>Preparing workspace…</Centered>;
  }

  return (
    <ThemeProvider theme={theme}>
      <GlobalStyle />
      {renderIntro && (
        <IntroOverlay $fadeOut={!showIntro} onClick={skipIntro}>
          <IntroBox $exiting={!showIntro}>
            <OrangeHint aria-hidden />
            <IntroFrame>
              <LogoWrapper>
                <LogoStill src={logoIntroStill} alt="STRAIGHT intro final frame" />
                {showGif && <LogoAnimation src={logoIntro} alt="STRAIGHT intro animation" />}
              </LogoWrapper>
            </IntroFrame>
          </IntroBox>
        </IntroOverlay>
      )}
      {screen}
    </ThemeProvider>
  );
}

export default App;

const IntroOverlay = styled.div<{ $fadeOut: boolean }>`
  position: fixed;
  inset: 0;
  display: grid;
  place-items: center;
  padding: ${({ theme }) => theme.spacing(3)};
  background:
    radial-gradient(circle at 18% 24%, rgba(255, 122, 26, 0.08), transparent 32%),
    radial-gradient(circle at 78% 18%, rgba(194, 81, 0, 0.08), transparent 36%),
    ${({ theme }) => theme.colors.background};
  z-index: 999;
  transition: opacity ${INTRO_FADE_MS}ms ease, visibility ${INTRO_FADE_MS}ms ease;
  pointer-events: ${({ $fadeOut }) => ($fadeOut ? 'none' : 'auto')};
  opacity: ${({ $fadeOut }) => ($fadeOut ? 0 : 1)};
  visibility: ${({ $fadeOut }) => ($fadeOut ? 'hidden' : 'visible')};
  cursor: pointer;
`;

const IntroBox = styled.div<{ $exiting: boolean }>`
  position: relative;
  width: min(860px, 94vw);
  padding: clamp(18px, 3vw, 32px);
  border-radius: ${({ theme }) => theme.radii.lg};
  background: linear-gradient(145deg, rgba(255, 255, 255, 0.94), rgba(255, 241, 224, 0.92));
  border: 1px solid rgba(255, 122, 26, 0.2);
  box-shadow: ${({ theme }) => theme.shadows.glow}, 0 28px 60px rgba(0, 0, 0, 0.08);
  overflow: hidden;
  backdrop-filter: blur(6px);
  isolation: isolate;
  transform: ${({ $exiting }) => ($exiting ? 'scale(1.48) translateY(-22px)' : 'scale(1) translateY(0)')};
  opacity: ${({ $exiting }) => ($exiting ? 0.56 : 1)};
  filter: ${({ $exiting }) => ($exiting ? 'blur(1px)' : 'none')};
  transition: transform 1100ms cubic-bezier(0.14, 0.62, 0.16, 1), opacity 1100ms ease, filter 1100ms ease;
`;

const OrangeHint = styled.div`
  position: absolute;
  inset: -16%;
  background: linear-gradient(120deg, transparent 42%, rgba(255, 122, 26, 0.18) 52%, transparent 62%);
  filter: blur(36px);
  transform: translateX(-8%);
  pointer-events: none;
  z-index: 0;
`;

const IntroFrame = styled.div`
  position: relative;
  z-index: 1;
  border-radius: calc(${({ theme }) => theme.radii.lg} - 8px);
  border: 1px solid rgba(46, 26, 10, 0.08);
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(255, 248, 240, 0.86));
  padding: clamp(22px, 3vw, 36px);
  display: grid;
  place-items: center;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.8);
`;

const LogoWrapper = styled.div`
  position: relative;
  display: grid;
  place-items: center;
  width: 100%;
  max-width: 720px;
  min-height: clamp(260px, 36vw, 440px);
`;

  const CompletionCard = styled.div`
  min-height: 70vh;
  display: grid;
  place-items: center;
  padding: ${({ theme }) => theme.spacing(5)};
  background:
    radial-gradient(circle at 16% 18%, rgba(255, 122, 26, 0.12), transparent 34%),
    radial-gradient(circle at 82% 12%, rgba(140, 103, 255, 0.14), transparent 36%),
    ${({ theme }) => theme.colors.background};
`;

const CompletionPanel = styled.div`
  width: min(640px, 100%);
  padding: ${({ theme }) => theme.spacing(4)};
  border-radius: ${({ theme }) => theme.radii.lg};
  background: linear-gradient(145deg, rgba(255, 255, 255, 0.96), rgba(255, 241, 224, 0.94));
  border: 1px solid rgba(255, 122, 26, 0.16);
  box-shadow: ${({ theme }) => theme.shadows.glow};
  display: grid;
  gap: ${({ theme }) => theme.spacing(2)};
`;

const CompletionTitle = styled.h2`
  margin: 0;
  font-family: ${({ theme }) => theme.fonts.heading};
  color: ${({ theme }) => theme.colors.text};
`;

const CompletionSubtitle = styled.p`
  margin: 0;
  color: ${({ theme }) => theme.colors.muted};
`;

const CompletionBadge = styled.div`
  display: inline-flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing(0.75)};
  padding: ${({ theme }) => theme.spacing(1)} ${({ theme }) => theme.spacing(1.5)};
  border-radius: 999px;
  background: rgba(255, 122, 26, 0.12);
  color: ${({ theme }) => theme.colors.neonAlt};
  font-weight: 700;
  font-size: 0.95rem;
  border: 1px solid rgba(255, 122, 26, 0.2);
  width: fit-content;
`;

const CompletionActions = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing(1.5)};
`;

const CompletionButton = styled.button`
  padding: ${({ theme }) => theme.spacing(1.5)} ${({ theme }) => theme.spacing(3)};
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid ${({ theme }) => theme.colors.neon};
  background: ${({ theme }) => theme.colors.neon};
  color: #05070f;
  font-weight: 700;
  cursor: pointer;
  box-shadow: ${({ theme }) => theme.shadows.glow};
  transition: transform ${({ theme }) => theme.durations.quick} ease, box-shadow ${({ theme }) => theme.durations.quick} ease;

  &:hover,
  &:focus-visible {
    transform: translateY(-2px);
    box-shadow: ${({ theme }) => theme.shadows.glow}, 0 12px 30px rgba(0, 0, 0, 0.08);
    outline: none;
  }
`;

const LogoAnimation = styled.img`
  width: min(90%, 680px);
  height: auto;
  object-fit: contain;
  position: absolute;
  inset: 0;
  margin: auto;
`;

const LogoStill = styled.img`
  width: min(90%, 680px);
  height: auto;
  object-fit: contain;
  display: block;
`;
