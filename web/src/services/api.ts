export type Profile = {
  name: string;
  drawWeight: string;
  notes: string;
};

export type PipelineStepStatus = 'pending' | 'active' | 'done';

export type PipelineProgress = {
  id: string;
  label: string;
  status: PipelineStepStatus;
};

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`Request to ${url} failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function getProfile(): Promise<Profile | null> {
  try {
    const profile = await fetchJson<Profile>('/api/personal-profile');
    return { ...profile, notes: profile.notes ?? '' };
  } catch (err) {
    console.error('Profile fetch failed.', err);
    return null;
  }
}

export async function saveProfile(profile: Profile): Promise<Profile> {
  const payload = { ...profile, notes: profile.notes ?? '' };
  const response = await fetch('/api/personal-profile', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`Profile save failed with status ${response.status}`);
  }
  const saved = (await response.json()) as Profile;
  return { ...saved, notes: saved.notes ?? '' };
}

export interface PipelineStatusResponse {
  jobId: string;
  steps: PipelineProgress[];
  done: boolean;
  error?: string;
}

export interface AnalysisAssets {
  original?: string;
  skeleton?: string;
  ghost?: string;
  spineImage?: string;
  drawForceImage?: string;
  drawForceFrames?: string[];
  drawLengthImage?: string;
  drawLengthFrames?: string[];
  releaseImage?: string;
  releaseFrames?: string[];
  followThroughImage?: string;
  followThroughFrames?: string[];
  postReleaseImage?: string;
  report?: string;
}

export interface AnalysisMarkdown {
  ghost?: string | null;
  spine?: string | null;
  drawForce?: string | null;
  drawLength?: string | null;
  postRelease?: string | null;
  release?: string | null;
  followThrough?: string | null;
  report?: string | null;
}

export type PhaseSegment = {
  name: string;
  startFrame: number;
  endFrame: number;
};

export interface AnalysisResult {
  jobId: string;
  videoId: string;
  assets: AnalysisAssets;
  markdown: AnalysisMarkdown;
  phases?: PhaseSegment[];
  frameCount?: number;
}

export type AnalysisCriterion = 'timing' | 'spine' | 'drawForce' | 'drawLength' | 'release' | 'followThrough';

export type AnalysisStatus = 'good' | 'bad' | 'unknown';

export type AnalysisSummary = {
  id: string;
  label: string;
  overview?: Partial<Record<AnalysisCriterion, AnalysisStatus>>;
};

export type RunManifestEntry = {
  id: string;
  label?: string;
  assets: AnalysisAssets;
  markdown?: Partial<AnalysisMarkdown>;
  phases?: PhaseSegment[];
  frameCount?: number;
  ghostMarkdown?: string;
  spineMarkdown?: string;
  drawForceMarkdown?: string;
  drawLengthMarkdown?: string;
  releaseMarkdown?: string;
  postReleaseMarkdown?: string;
  postReleaseDrawMarkdown?: string;
  postReleaseBowMarkdown?: string;
  followThroughMarkdown?: string;
};

type RunsManifest = {
  runs: RunManifestEntry[];
};

const RUN_MANIFEST_PATH = '/assets/runs/index.json';
const MIN_DATASET_SAMPLES = 5;

const toAssetUrl = (path?: string): string | undefined => {
  if (!path) return undefined;
  if (/^https?:\/\//i.test(path)) return path;
  const cleaned = path.replace(/^\//, '');
  return cleaned.startsWith('assets/') ? `/${cleaned}` : `/assets/${cleaned}`;
};

async function fetchRunsManifest(): Promise<RunsManifest | null> {
  try {
    return await fetchJson<RunsManifest>(RUN_MANIFEST_PATH);
  } catch (err) {
    console.warn('Runs manifest not available in static assets.', err);
    return null;
  }
}

async function fetchTextAsset(assetPath?: string): Promise<string | null> {
  const assetUrl = toAssetUrl(assetPath);
  if (!assetUrl) return null;
  try {
    const response = await fetch(assetUrl);
    if (!response.ok) {
      throw new Error(`Asset fetch failed with status ${response.status}`);
    }
    return response.text();
  } catch (err) {
    console.warn(`Unable to load text asset from ${assetUrl}`, err);
    return null;
  }
}

const fetchRunReport = (reportPath?: string) => fetchTextAsset(reportPath);

const stripImageMarkdown = (markdown: string) => markdown.replace(/!\[[^\]]*]\([^)]+\)/g, '').trim();

const fetchAssetDate = async (assetPath?: string): Promise<Date | null> => {
  const assetUrl = toAssetUrl(assetPath);
  if (!assetUrl) return null;
  try {
    const response = await fetch(assetUrl, { method: 'HEAD' });
    const lastModified = response.headers.get('last-modified');
    if (lastModified) {
      const parsed = new Date(lastModified);
      if (!Number.isNaN(parsed.getTime())) {
        return parsed;
      }
    }
  } catch (err) {
    console.warn(`Unable to resolve date for ${assetUrl}`, err);
  }
  return null;
};

const formatRunLabel = (date: Date, sequence: number): string => {
  const month = date.toLocaleString('en-US', { month: 'short' });
  const day = date.getDate();
  const year = String(date.getFullYear()).slice(-2);
  return `${month}. ${day}, '${year} #${sequence}`;
};

const resolveRunCreatedAt = async (run: RunManifestEntry): Promise<Date | null> => {
  return (
    (await fetchAssetDate(run.assets.original)) ||
    (await fetchAssetDate(run.assets.report)) ||
    (await fetchAssetDate(run.assets.skeleton)) ||
    (await fetchAssetDate(run.assets.ghost)) ||
    null
  );
};

function buildAssetGallery(assets: AnalysisAssets): string {
  const images: Array<[string, string | undefined]> = [
    ['Spine straightness', assets.spineImage],
    ['Draw-force line', assets.drawForceImage],
    ['Draw length', assets.drawLengthImage ?? assets.drawForceImage],
    ['Release', assets.releaseImage ?? assets.postReleaseImage],
    ['Follow-through', assets.followThroughImage ?? assets.postReleaseImage],
    ['Post-release follow-through', assets.postReleaseImage],
  ];
  const gallery = images
    .map(([label, path]) => {
      const assetPath = toAssetUrl(path);
      return assetPath ? `![${label}](${assetPath})` : null;
    })
    .filter((item): item is string => Boolean(item));
  if (!gallery.length) return '';
  return `\n\n---\nAssets\n\n${gallery.join('\n\n')}`;
}

const normalizePhases = (phases?: PhaseSegment[]): PhaseSegment[] => {
  if (!phases?.length) return [];
  return phases
    .map((phase) => {
      const rawStart = (phase as { start_frame?: number; startFrame?: number }).startFrame ?? (phase as { start_frame?: number }).start_frame ?? 0;
      const rawEnd = (phase as { end_frame?: number; endFrame?: number }).endFrame ?? (phase as { end_frame?: number }).end_frame ?? rawStart;
      const startFrame = Number.isFinite(rawStart) ? Number(rawStart) : 0;
      const endFrame = Number.isFinite(rawEnd) ? Number(rawEnd) : startFrame;
      const name = phase.name || 'phase';
      return { name, startFrame, endFrame };
    })
    .filter((phase) => phase.name.length > 0);
};

const inferFrameCount = (phases: PhaseSegment[], fallback?: number): number | undefined => {
  if (fallback && fallback > 0) return fallback;
  const maxEnd = phases.reduce((max, phase) => Math.max(max, phase.endFrame), 0);
  return maxEnd > 0 ? maxEnd : undefined;
};

function buildRunMarkdown(run: RunManifestEntry, reportText: string | null, requestedId: string): AnalysisMarkdown {
  const resolvedReport = run.markdown?.report ?? reportText;
  const cleanedReport = resolvedReport ? stripImageMarkdown(resolvedReport) : '';
  const gallery = buildAssetGallery(run.assets);
  const mismatchNote =
    requestedId && requestedId !== run.id
      ? `_Requested analysis "${requestedId}" not found in /assets/runs; showing "${run.label ?? run.id}"._\n\n`
      : '';

  let report: string | null = null;
  if (cleanedReport || gallery || mismatchNote) {
    const body = cleanedReport || 'Full report not available in static assets.';
    report = `${mismatchNote}${body}${gallery}`.trim();
  }

  return {
    ghost: run.markdown?.ghost ?? 'Ghost overlay is ready in the video player.',
    spine:
      run.markdown?.spine ??
      (run.assets.spineImage ? 'Spine straightness snapshot from this run.' : 'Spine visualization missing from assets.'),
    drawForce:
      run.markdown?.drawForce ??
      (run.assets.drawForceImage ? 'Draw-force line overlay from this run.' : 'Draw-force visualization missing from assets.'),
    drawLength:
      run.markdown?.drawLength ??
      (run.assets.drawForceImage ? 'Draw length analysis for this run.' : 'Draw length visualization missing from assets.'),
    release: run.markdown?.release ?? 'Release follow-through analysis unavailable.',
    followThrough: run.markdown?.followThrough ?? 'Follow-through stability analysis unavailable.',
    postRelease:
      run.markdown?.postRelease ??
      (run.assets.postReleaseImage ? 'Post-release follow-through visualization.' : 'Post-release visualization missing from assets.'),
    report,
  };
}

export async function uploadTrainingVideo(_file: File, _useStPrompt = false): Promise<{ uploadId: string }> {
  const formData = new FormData();
  formData.append('video', _file);
  formData.append('useStPrompt', String(_useStPrompt));

  try {
    const response = await fetch('/api/upload', { method: 'POST', body: formData });
    if (!response.ok) {
      throw new Error(`Upload failed with status ${response.status}`);
    }
    const data = (await response.json()) as { uploadId?: string; jobId?: string };
    const uploadId = data.uploadId ?? data.jobId;
    if (!uploadId) {
      throw new Error('Upload response missing uploadId');
    }
    return { uploadId };
  } catch (err) {
    console.warn('Upload failed, falling back to static stub.', err);
    return { uploadId: `local-${Date.now()}` };
  }
}

export async function pollPipelineStatus(uploadId: string): Promise<PipelineStatusResponse> {
  try {
    return await fetchJson<PipelineStatusResponse>(`/api/pipeline-status/${uploadId}`);
  } catch (err) {
    console.warn('Pipeline status fallback to static steps.', err);
    const steps = defaultPipelineSteps().map((step) => ({ ...step, status: 'done' as const }));
    return { jobId: uploadId, steps, done: true };
  }
}

export function defaultPipelineSteps(): PipelineProgress[] {
  return [
    { id: 'transcode', label: 'Transcoding to 60 fps', status: 'pending' },
    { id: 'upload', label: 'Uploading video', status: 'pending' },
    { id: 'remote-inference', label: 'Remote inference', status: 'pending' },
    { id: 'phase-estimation', label: 'Phase estimation', status: 'pending' },
    { id: 'kalman', label: 'Kalman smoothing', status: 'pending' },
    { id: 'crop', label: 'Cropping & bannering', status: 'pending' },
    { id: 'skeleton', label: 'Skeleton overlay', status: 'pending' },
    { id: 'ghost', label: 'Ghost overlay analysis', status: 'pending' },
    { id: 'draw-force', label: 'Draw-force analysis', status: 'pending' },
    { id: 'post-release', label: 'Post-release analysis', status: 'pending' },
    { id: 'finalize', label: 'Finalizing report', status: 'pending' },
  ];
}

export async function fetchAnalysisResults(jobId: string): Promise<AnalysisResult> {
  try {
    const live = await fetchJson<{ jobId: string; videoId: string; assets: AnalysisAssets; markdown: Partial<AnalysisMarkdown> }>(
      `/api/jobs/${jobId}/results`,
    );
    return {
      jobId: live.jobId ?? jobId,
      videoId: live.videoId ?? jobId,
      assets: live.assets ?? {},
      markdown: {
        ghost: live.markdown?.ghost ?? undefined,
        spine: live.markdown?.spine ?? undefined,
        drawForce: live.markdown?.drawForce ?? undefined,
        drawLength: live.markdown?.drawLength ?? undefined,
        release: live.markdown?.postRelease ?? live.markdown?.release ?? undefined,
        followThrough: live.markdown?.postRelease ?? live.markdown?.followThrough ?? undefined,
        postRelease: live.markdown?.postRelease ?? undefined,
        report: live.markdown?.report ?? undefined,
      },
      phases: [],
    };
  } catch (err) {
    console.warn('Live job results unavailable; falling back to static assets.', err);
  }

  const manifest = await fetchRunsManifest();
  if (manifest?.runs?.length) {
    const baseRun = manifest.runs.find((entry) => entry.id === jobId) ?? manifest.runs[0];
    if (baseRun.id !== jobId) {
      console.warn(`Analysis id "${jobId}" not found in runs manifest; using "${baseRun.id}" instead.`);
    }
    const reportText = await fetchRunReport(baseRun.assets.report);
    const ghostText = await fetchTextAsset(baseRun.ghostMarkdown);
    const spineText = await fetchTextAsset(baseRun.spineMarkdown);
    const drawForceText = await fetchTextAsset(baseRun.drawForceMarkdown);
    const drawLengthText = await fetchTextAsset(baseRun.drawLengthMarkdown);
    const releaseText = await fetchTextAsset(baseRun.releaseMarkdown);
    const followThroughText = await fetchTextAsset(baseRun.followThroughMarkdown);
    const phases = normalizePhases(baseRun.phases);
    const frameCount = inferFrameCount(phases, baseRun.frameCount);
    const run: RunManifestEntry = {
      ...baseRun,
      markdown: {
        ...baseRun.markdown,
        ghost: ghostText ?? baseRun.markdown?.ghost,
        spine: spineText ?? baseRun.markdown?.spine,
        drawForce: drawForceText ?? baseRun.markdown?.drawForce,
        drawLength: drawLengthText ?? baseRun.markdown?.drawLength,
        release: releaseText ?? baseRun.markdown?.release,
        followThrough: followThroughText ?? baseRun.markdown?.followThrough,
      },
    };
    return {
      jobId: run.id,
      videoId: run.id,
      assets: run.assets,
      markdown: buildRunMarkdown(run, reportText, jobId),
      phases,
      frameCount,
    };
  }

  try {
    const data = await fetchJson<AnalysisResult>('/assets/analysis.json');
    return {
      ...data,
      jobId: data.jobId ?? jobId,
      videoId: data.videoId ?? jobId,
      phases: data.phases ? normalizePhases(data.phases) : data.phases,
      frameCount: inferFrameCount(data.phases ?? [], data.frameCount),
    };
  } catch (err) {
    console.warn('Static analysis file missing; returning placeholder data.', err);
    return {
      jobId,
      videoId: jobId,
      assets: {},
      markdown: {
        ghost: 'Ghost overlay results unavailable. Add a run entry to web/public/assets/runs/index.json.',
        spine: 'Spine analysis unavailable. Add a run entry to web/public/assets/runs/index.json.',
        drawForce: 'Draw-force analysis unavailable. Add a run entry to web/public/assets/runs/index.json.',
        drawLength: 'Draw length analysis unavailable. Add a run entry to web/public/assets/runs/index.json.',
        release: 'Release follow-through analysis unavailable. Add a run entry to web/public/assets/runs/index.json.',
        followThrough: 'Follow-through stability analysis unavailable. Add a run entry to web/public/assets/runs/index.json.',
        postRelease: 'Post-release metrics unavailable. Add a run entry to web/public/assets/runs/index.json.',
      },
    };
  }
}

const toNumber = (val?: string | number | null): number | undefined => {
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

const parseDataBlock = (markdown?: string | null): Record<string, number | string> => {
  if (!markdown) return {};
  const lines = markdown.split('\n');
  const startIdx = lines.findIndex((line) => line.trim() === 'DATA_START');
  const endIdx = lines.findIndex((line, idx) => idx > startIdx && line.trim() === 'DATA_END');
  if (startIdx === -1 || endIdx === -1 || endIdx <= startIdx) return {};
  const data: Record<string, number | string> = {};
  lines.slice(startIdx + 1, endIdx).forEach((line) => {
    const match = line.match(/^\s*([^=]+)=(.+)$/);
    if (!match) return;
    const [, key, rawVal] = match;
    const maybeNum = toNumber(rawVal.trim());
    data[key.trim()] = maybeNum !== undefined ? maybeNum : rawVal.trim();
  });
  return data;
};

const statusFromWindow = (value?: number, avg?: number, std?: number, grace = 0): AnalysisStatus => {
  if (value === undefined || avg === undefined || std === undefined || std <= 0) return 'unknown';
  const diff = Math.abs(value - avg);
  return diff <= std + grace ? 'good' : 'bad';
};

const coalesceStatus = (...statuses: AnalysisStatus[]): AnalysisStatus => statuses.find((s) => s !== 'unknown') ?? 'unknown';

const statusFromMarkdown = (markdown?: string | null): AnalysisStatus => {
  if (!markdown) return 'unknown';
  const timingMatch = markdown.match(/timing_status\s*=\s*([^\s]+)/i);
  if (timingMatch) {
    const value = timingMatch[1].trim().toLowerCase();
    if (value === 'good') return 'good';
    if (value === 'bad') return 'bad';
    // Let textual cues decide when the structured status is unknown or unrecognized.
  }
  if (/\bgood\b/i.test(markdown)) return 'good';
  if (/(needs attention|warning|issue|poor|bad|too short|too_long|too-short)/i.test(markdown)) return 'bad';
  return 'unknown';
};

async function buildOverview(run: RunManifestEntry): Promise<AnalysisSummary['overview']> {
  const [ghost, spine, drawForce, drawLength, release, followThrough] = await Promise.all([
    fetchTextAsset(run.ghostMarkdown),
    fetchTextAsset(run.spineMarkdown),
    fetchTextAsset(run.drawForceMarkdown),
    fetchTextAsset(run.drawLengthMarkdown),
    fetchTextAsset(run.releaseMarkdown ?? run.postReleaseMarkdown ?? run.postReleaseDrawMarkdown),
    fetchTextAsset(run.followThroughMarkdown ?? run.postReleaseBowMarkdown),
  ]);

  const ghostData = parseDataBlock(ghost);
  const drawForceData = parseDataBlock(drawForce);
  const drawLengthData = parseDataBlock(drawLength);
  const releaseData = parseDataBlock(release);
  const followData = parseDataBlock(followThrough);

  const timingSamples = toNumber(ghostData.dataset_draw_to_release_sample_count);
  const timingMinSamples = toNumber(ghostData.dataset_min_samples_required) ?? MIN_DATASET_SAMPLES;
  const timingStd = toNumber(ghostData.dataset_draw_to_release_std_seconds);
  const hasTimingWindow =
    timingStd !== undefined && timingStd > 0 && (timingSamples === undefined || timingSamples >= timingMinSamples);
  const timingStatus = hasTimingWindow
    ? coalesceStatus(
        statusFromWindow(
          toNumber(ghostData.draw_to_release_seconds),
          toNumber(ghostData.dataset_draw_to_release_avg_seconds),
          timingStd,
        ),
        statusFromMarkdown(ghost),
      )
    : 'unknown';

  const drawForceSamples = toNumber(drawForceData.draw_force_angle_sample_count);
  const drawForceMinSamples = toNumber(drawForceData.draw_force_min_samples_required) ?? MIN_DATASET_SAMPLES;
  const drawForceStd = toNumber(drawForceData.draw_force_angle_std_deg);
  const hasDrawForceWindow =
    (drawForceSamples === undefined || drawForceSamples >= drawForceMinSamples) &&
    drawForceStd !== undefined &&
    drawForceStd > 0;
  const drawForceStatus = hasDrawForceWindow
    ? coalesceStatus(
        statusFromWindow(
          toNumber(drawForceData.draw_force_angle_deg),
          toNumber(drawForceData.draw_force_angle_avg_deg),
          drawForceStd,
        ),
        statusFromMarkdown(drawForce),
      )
    : 'unknown';

  const drawLengthSamples = toNumber(drawLengthData.draw_length_sample_count);
  const drawLengthMinSamples = toNumber(drawLengthData.draw_length_min_samples_required) ?? MIN_DATASET_SAMPLES;
  const drawLengthStd = toNumber(drawLengthData.draw_length_std_hipwidths);
  const hasDrawLengthWindow =
    (drawLengthSamples === undefined || drawLengthSamples >= drawLengthMinSamples) &&
    drawLengthStd !== undefined &&
    drawLengthStd > 0;
  const drawLengthStatus = hasDrawLengthWindow
    ? coalesceStatus(
        statusFromWindow(
          toNumber(drawLengthData.draw_length_hipwidths),
          toNumber(drawLengthData.draw_length_avg_hipwidths),
          drawLengthStd,
          0.1,
        ),
        statusFromMarkdown(drawLength),
      )
    : 'unknown';

  const releaseChangePct = toNumber(releaseData.nose_draw_length_change_pct);
  const releaseThreshold = toNumber(releaseData.nose_draw_length_good_threshold_pct);
  const releaseStatus =
    releaseChangePct !== undefined && releaseThreshold !== undefined
      ? releaseChangePct > releaseThreshold
        ? 'good'
        : 'bad'
      : statusFromMarkdown(release);

  const followDelta = toNumber(followData.bow_torso_angle_delta_deg);
  const followThreshold = toNumber(followData.bow_torso_angle_threshold_deg);
  const followStatus =
    followDelta !== undefined && followThreshold !== undefined
      ? followDelta < followThreshold
        ? 'good'
        : 'bad'
      : statusFromMarkdown(followThrough);

  return {
    timing: timingStatus,
    spine: statusFromMarkdown(spine),
    drawForce: drawForceStatus,
    drawLength: drawLengthStatus,
    release: releaseStatus,
    followThrough: followStatus,
  };
}

export async function fetchAvailableAnalyses(): Promise<AnalysisSummary[]> {
  const manifest = await fetchRunsManifest();
  if (manifest?.runs?.length) {
    const runsWithDate = await Promise.all(
      manifest.runs.map(async (run, index) => ({
        run,
        index,
        createdAt: await resolveRunCreatedAt(run),
      })),
    );

    const sortedForSequencing = [...runsWithDate].sort((a, b) => {
      const aTime = a.createdAt?.getTime();
      const bTime = b.createdAt?.getTime();
      if (aTime !== undefined && bTime !== undefined && !Number.isNaN(aTime) && !Number.isNaN(bTime)) {
        return aTime - bTime;
      }
      if (aTime !== undefined && !Number.isNaN(aTime)) return -1;
      if (bTime !== undefined && !Number.isNaN(bTime)) return 1;
      return a.index - b.index;
    });

    const sequences = new Map<string, { createdAt: Date | null; sequence: number }>();
    const dayCounters = new Map<string, number>();
    sortedForSequencing.forEach(({ run, createdAt }) => {
      const dayKey = createdAt ? createdAt.toISOString().slice(0, 10) : 'unknown';
      const seq = (dayCounters.get(dayKey) ?? 0) + 1;
      dayCounters.set(dayKey, seq);
      sequences.set(run.id, { createdAt, sequence: seq });
    });

    const entries = await Promise.all(
      manifest.runs.map(async (run) => {
        const meta = sequences.get(run.id);
        const label =
          meta?.createdAt && meta.sequence
            ? formatRunLabel(meta.createdAt, meta.sequence)
            : run.label ?? run.id;
        return {
          id: run.id,
          label,
          overview: await buildOverview(run),
          createdAt: meta?.createdAt?.getTime() ?? null,
          sequence: meta?.sequence ?? 0,
        };
      }),
    );

    return entries
      .sort((a, b) => {
        if (a.createdAt !== null && b.createdAt !== null && a.createdAt !== b.createdAt) {
          return b.createdAt - a.createdAt;
        }
        return b.sequence - a.sequence;
      })
      .map(({ createdAt: _createdAt, sequence: _sequence, ...rest }) => rest);
  }

  try {
    const data = await fetchJson<AnalysisResult>('/assets/analysis.json');
    const id = data.jobId ?? 'analysis';
    return [{ id, label: id }];
  } catch (err) {
    console.warn('No analyses available in static assets.', err);
    return [];
  }
}

export type RegenerateReportsResult = { runId: string; generated: string[]; warnings?: string[] };

export async function regenerateReports(runId: string): Promise<RegenerateReportsResult> {
  const endpoints = [`/api/runs/${runId}/reports/regenerate`, `/api/runs/${runId}/missing-reports`];
  let lastStatus: number | null = null;
  for (const url of endpoints) {
    const response = await fetch(url, { method: 'POST' });
    lastStatus = response.status;
    if (response.ok) {
      return response.json();
    }
    // Try the next endpoint only if this one was not found (backwards compatibility)
    if (response.status !== 404) {
      break;
    }
  }
  throw new Error(`Report regeneration failed${lastStatus ? ` with status ${lastStatus}` : ''}`);
}

export const runMissingReports = regenerateReports;

export type CoachChatMessage = { role: 'user' | 'assistant' | 'coach'; content: string };

export type CoachChatResponse = {
  reply: string;
  latestRunId?: string | null;
  contextPreview?: string | null;
};

export async function sendCoachChat(
  message: string,
  history: CoachChatMessage[] = [],
  runId?: string | null,
): Promise<CoachChatResponse> {
  const response = await fetch('/api/coach-chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history, runId }),
  });
  if (!response.ok) {
    const detail = await response.text().catch(() => '');
    throw new Error(detail || `Coach chat failed with status ${response.status}`);
  }
  return response.json();
}

export type RunExclusions = { excluded: string[] };

export async function getRunExclusions(): Promise<RunExclusions> {
  return fetchJson<RunExclusions>('/api/run-exclusions');
}

export async function saveRunExclusions(excluded: string[]): Promise<RunExclusions> {
  const response = await fetch('/api/run-exclusions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ excluded }),
  });
  if (!response.ok) {
    throw new Error(`Failed to save run exclusions (${response.status})`);
  }
  return response.json();
}

export async function getGhostReference(): Promise<{ referenceId: string }> {
  const response = await fetch('/api/ghost-reference');
  if (!response.ok) {
    throw new Error(`Ghost reference fetch failed with status ${response.status}`);
  }
  return response.json();
}

export async function setGhostReference(referenceId: string): Promise<{ referenceId: string }> {
  const response = await fetch('/api/ghost-reference', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ referenceId }),
  });
  if (!response.ok) {
    throw new Error(`Ghost reference update failed with status ${response.status}`);
  }
  return response.json();
}
