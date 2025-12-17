import { useEffect, useState } from 'react';
import styled, { css, keyframes } from 'styled-components';
import logoBadge from '../assets/tpn_logo.png';
import {
  fetchAvailableAnalyses,
  type AnalysisSummary,
  type AnalysisStatus,
  type AnalysisCriterion,
  type RegenerateReportsResult,
  getRunExclusions,
  saveRunExclusions,
} from '../services/api';

const Container = styled.section`
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing(4)};
  padding: ${({ theme }) => theme.spacing(6)};
  background: radial-gradient(circle at top left, rgba(140, 103, 255, 0.15), transparent 40%),
    ${({ theme }) => theme.colors.background};
`;

const HeaderBar = styled.header`
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

const Title = styled.h1`
  margin: 0;
  font-family: ${({ theme }) => theme.fonts.heading};
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing(1)};
`;

const Subtitle = styled.p`
  margin: 0;
  color: ${({ theme }) => theme.colors.muted};
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

const UploadButton = styled.div`
  align-self: flex-start;
  padding: ${({ theme }) => theme.spacing(1)} ${({ theme }) => theme.spacing(3)};
  border-radius: ${({ theme }) => theme.radii.md};
  background: linear-gradient(120deg, ${({ theme }) => theme.colors.neon} 0%, rgba(255, 200, 150, 0.92) 100%);
  color: #120c06;
  font-weight: 700;
  box-shadow: 0 12px 30px rgba(255, 122, 26, 0.22);
  transition: transform ${({ theme }) => theme.durations.quick} ease, box-shadow ${({ theme }) =>
      theme.durations.quick} ease, background ${({ theme }) => theme.durations.quick} ease;

  &:hover,
  &:focus-visible {
    transform: translateY(-2px);
    background: linear-gradient(120deg, rgba(255, 170, 120, 0.98) 0%, rgba(255, 236, 214, 0.95) 100%);
    box-shadow: 0 16px 38px rgba(255, 170, 120, 0.3);
    outline: none;
  }
`;

const HiddenInput = styled.input`
  position: absolute;
  opacity: 0;
  pointer-events: none;
`;

const UploadGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: ${({ theme }) => theme.spacing(2.5)};
  align-items: stretch;

  @media (max-width: 900px) {
    grid-template-columns: 1fr;
  }
`;

const UploadNote = styled.label`
  position: relative;
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing(1.5)};
  padding: ${({ theme }) => theme.spacing(3.5)};
  min-height: 240px;
  border-radius: ${({ theme }) => theme.radii.lg};
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(255, 241, 224, 0.92));
  border: 1px dashed rgba(255, 122, 26, 0.32);
  box-shadow: ${({ theme }) => theme.shadows.glow}, 0 18px 40px rgba(0, 0, 0, 0.08);
  cursor: pointer;
  overflow: hidden;
  isolation: isolate;
  transition: transform ${({ theme }) => theme.durations.quick} ease, box-shadow ${({ theme }) => theme.durations.quick} ease;

  &:hover,
  &:focus-within {
    transform: translateY(-3px);
    box-shadow: ${({ theme }) => theme.shadows.glow}, 0 24px 50px rgba(0, 0, 0, 0.1);
  }
`;

const TipsNote = styled.div`
  position: relative;
  display: grid;
  grid-template-columns: 160px 1fr;
  gap: ${({ theme }) => theme.spacing(1.5)};
  padding: ${({ theme }) => theme.spacing(3.5)};
  min-height: 240px;
  border-radius: ${({ theme }) => theme.radii.lg};
  background: linear-gradient(150deg, rgba(255, 248, 240, 0.96), rgba(255, 255, 255, 0.94));
  border: 1px solid rgba(0, 0, 0, 0.04);
  box-shadow: 0 18px 40px rgba(0, 0, 0, 0.08);
  align-items: center;
  overflow: hidden;
  isolation: isolate;

  @media (max-width: 760px) {
    grid-template-columns: 1fr;
  }
`;

const ExampleFrame = styled.div`
  position: relative;
  width: 100%;
  max-width: 210px;
  padding: ${({ theme }) => theme.spacing(0.5)};
  border-radius: ${({ theme }) => theme.radii.md};
  background: linear-gradient(145deg, rgba(255, 241, 224, 0.88), rgba(255, 255, 255, 0.9));
  border: 1px solid rgba(0, 0, 0, 0.04);
  box-shadow: 0 14px 32px rgba(0, 0, 0, 0.08);
  overflow: hidden;
  justify-self: center;

  &::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(120deg, rgba(255, 170, 120, 0.08), rgba(255, 255, 255, 0.06));
    pointer-events: none;
  }
`;

const ExampleImage = styled.img`
  position: relative;
  display: block;
  width: 100%;
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid rgba(0, 0, 0, 0.05);
`;

const TipTitle = styled.h4`
  margin: 0;
  font-size: 1rem;
  color: ${({ theme }) => theme.colors.text};
`;

const TipBody = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing(1)};
`;

const StepList = styled.ol`
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: ${({ theme }) => theme.spacing(0.75)};
`;

const StepItem = styled.li`
  display: grid;
  grid-template-columns: auto 1fr;
  gap: ${({ theme }) => theme.spacing(1)};
  align-items: start;
`;

const StepNum = styled.span`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: rgba(255, 122, 26, 0.12);
  color: ${({ theme }) => theme.colors.neonAlt};
  font-weight: 700;
  font-size: 0.9rem;
  border: 1px solid rgba(255, 122, 26, 0.28);
`;

const StepText = styled.span`
  color: ${({ theme }) => theme.colors.text};
  line-height: 1.4;
`;

const LogoBadge = styled.img`
  height: 40px;
  width: auto;
  display: block;
`;

const BrandText = styled.span`
  font-weight: 700;
  letter-spacing: 0.03em;
`;

const IconSvg = styled.svg`
  width: 22px;
  height: 22px;
  fill: currentColor;
`;

const AnalysesSection = styled.div`
  margin-top: ${({ theme }) => theme.spacing(2)};
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing(1)};
`;

const CoachCard = styled.div`
  margin-top: ${({ theme }) => theme.spacing(2)};
  padding: ${({ theme }) => theme.spacing(3)};
  border-radius: ${({ theme }) => theme.radii.lg};
  background: linear-gradient(140deg, rgba(255, 248, 240, 0.95), rgba(255, 255, 255, 0.94));
  border: 1px solid rgba(255, 122, 26, 0.18);
  box-shadow: ${({ theme }) => theme.shadows.glow}, 0 14px 32px rgba(0, 0, 0, 0.06);
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: ${({ theme }) => theme.spacing(2)};
  justify-content: space-between;
`;

const CoachMeta = styled.div`
  display: inline-flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing(1.5)};
`;

const CoachBadge = styled.img`
  width: 72px;
  height: 72px;
  border-radius: ${({ theme }) => theme.radii.md};
  object-fit: contain;
  background: none;
  border: none;
  box-shadow: none;
  padding: 0;
`;

const CoachText = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing(0.5)};
  color: ${({ theme }) => theme.colors.text};
`;

const CoachTitle = styled.h3`
  margin: 0;
  font-family: ${({ theme }) => theme.fonts.heading};
  color: ${({ theme }) => theme.colors.text};
`;

const CoachSubtitle = styled.p`
  margin: 0;
  color: ${({ theme }) => theme.colors.muted};
`;

const CoachButton = styled.button`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: ${({ theme }) => theme.spacing(0.75)};
  padding: ${({ theme }) => theme.spacing(1.25)} ${({ theme }) => theme.spacing(2)};
  border-radius: ${({ theme }) => theme.radii.md};
  background: linear-gradient(120deg, #ffd5c2, #ff9f7a);
  color: #05070f;
  font-weight: 700;
  border: 1px solid rgba(255, 159, 122, 0.6);
  box-shadow: ${({ theme }) => theme.shadows.glow};
  transition: transform ${({ theme }) => theme.durations.quick} ease, box-shadow ${({ theme }) => theme.durations.quick} ease,
    filter ${({ theme }) => theme.durations.quick} ease;
  cursor: pointer;
  text-decoration: none;
  outline: none;

  &:hover,
  &:focus-visible {
    transform: translateY(-1px);
    box-shadow: ${({ theme }) => theme.shadows.glow}, 0 12px 30px rgba(0, 0, 0, 0.08);
    outline: none;
    filter: brightness(1.05);
  }
`;

const AnalysesHeading = styled.h3`
  margin: 0;
  font-size: 1rem;
  color: ${({ theme }) => theme.colors.text};
`;

const AnalysesList = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing(1)};
`;

const ExclusionPanel = styled.div`
  background: ${({ theme }) => theme.colors.surface};
  border-radius: ${({ theme }) => theme.radii.lg};
  border: 1px solid rgba(0, 0, 0, 0.06);
  box-shadow: ${({ theme }) => theme.shadows.glow}, 0 12px 32px rgba(0, 0, 0, 0.08);
  padding: ${({ theme }) => theme.spacing(3)};
  max-width: 640px;
  width: min(640px, 90vw);
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing(2)};
  max-height: 80vh;
  overflow: hidden;
`;

const ExclusionHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: ${({ theme }) => theme.spacing(1)};
`;

const ExclusionList = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: ${({ theme }) => theme.spacing(1)};
  max-height: 50vh;
  overflow-y: auto;
  padding-right: ${({ theme }) => theme.spacing(1)};
`;

const ExclusionItem = styled.label`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing(1)};
  padding: ${({ theme }) => theme.spacing(1)};
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid rgba(0, 0, 0, 0.08);
  background: ${({ theme }) => theme.colors.surfaceAlt};
`;

const ExclusionActions = styled.div`
  display: flex;
  justify-content: flex-end;
  gap: ${({ theme }) => theme.spacing(1)};
  position: sticky;
  bottom: 0;
  padding-top: ${({ theme }) => theme.spacing(1)};
  background: ${({ theme }) => theme.colors.surface};
`;

const AnalysisItem = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing(1.25)};
  padding: ${({ theme }) => theme.spacing(2)};
  border-radius: ${({ theme }) => theme.radii.lg};
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.96), rgba(255, 241, 224, 0.9));
  border: 1px solid rgba(255, 122, 26, 0.16);
  box-shadow: ${({ theme }) => theme.shadows.glow};
`;

const TitleRow = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: ${({ theme }) => theme.spacing(2)};
`;

const AnalysisLabel = styled.span`
  font-weight: 700;
  font-size: 1.05rem;
  color: ${({ theme }) => theme.colors.text};
`;

const ViewButton = styled.button`
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 18px;
  border-radius: 999px;
  border: 1px solid rgba(255, 122, 26, 0.28);
  background: linear-gradient(120deg, rgba(255, 245, 234, 0.95), rgba(255, 224, 192, 0.9));
  color: ${({ theme }) => theme.colors.text};
  cursor: pointer;
  font-weight: 700;
  letter-spacing: 0.01em;
  box-shadow: 0 12px 28px rgba(0, 0, 0, 0.08);
  transition: transform ${({ theme }) => theme.durations.quick} ease, box-shadow ${({ theme }) =>
      theme.durations.quick} ease, background ${({ theme }) => theme.durations.quick} ease,
    border-color ${({ theme }) => theme.durations.quick} ease;

  &::after {
    content: '→';
    font-size: 0.95rem;
    color: ${({ theme }) => theme.colors.neonAlt};
    transition: transform ${({ theme }) => theme.durations.quick} ease;
  }

  &:hover,
  &:focus-visible {
    outline: none;
    transform: translateY(-1px);
    box-shadow: ${({ theme }) => theme.shadows.glow}, 0 18px 32px rgba(0, 0, 0, 0.08);
    background: linear-gradient(120deg, rgba(255, 236, 215, 0.98), rgba(255, 215, 170, 0.94));
    border-color: rgba(255, 122, 26, 0.42);

    &::after {
      transform: translateX(2px);
    }
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    box-shadow: none;
    transform: none;
  }
`;

const SecondaryButton = styled(ViewButton)`
  background: linear-gradient(135deg, ${({ theme }) => theme.colors.surfaceAlt}, ${({ theme }) => theme.colors.surface});
  border-color: rgba(0, 0, 0, 0.08);
  box-shadow: 0 10px 22px rgba(0, 0, 0, 0.08);

  &::after {
    display: none;
  }
`;

const ModalBackdrop = styled.div`
  position: fixed;
  inset: 0;
  display: grid;
  place-items: center;
  background: rgba(0, 0, 0, 0.32);
  backdrop-filter: blur(3px);
  z-index: 80;
  padding: ${({ theme }) => theme.spacing(3)};
`;

const ModalCard = styled.div<{ $tone: 'success' | 'warn' }>`
  width: min(520px, 96vw);
  padding: ${({ theme }) => theme.spacing(3)};
  border-radius: 28px;
  background: ${({ theme }) =>
    `radial-gradient(circle at 24% 22%, rgba(255, 122, 26, 0.16), transparent 38%), linear-gradient(145deg, ${theme.colors.surface}, ${theme.colors.surfaceAlt})`};
  border: 1px solid ${({ $tone }) => ($tone === 'success' ? 'rgba(255, 122, 26, 0.36)' : 'rgba(255, 122, 26, 0.36)')};
  box-shadow: ${({ theme }) => theme.shadows.glow}, 0 22px 48px rgba(0, 0, 0, 0.18);
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing(2)};
  text-align: center;
`;

const ModalMessage = styled.p`
  margin: 0;
  color: ${({ theme }) => theme.colors.text};
  line-height: 1.5;
  font-weight: 600;
`;

const ModalActions = styled.div`
  display: flex;
  justify-content: center;
  gap: ${({ theme }) => theme.spacing(1)};
`;

const ModalButton = styled(ViewButton)`
  min-width: 120px;

  &::after {
    display: none;
  }

  align-self: center;
  justify-content: center;
`;

const CriteriaGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 10px 14px;
  margin-top: 12px;
  padding: 12px 12px 6px;
  border-radius: ${({ theme }) => theme.radii.md};
  background: rgba(255, 255, 255, 0.7);
  border: 1px dashed rgba(255, 122, 26, 0.22);
`;

const CriteriaItem = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  color: ${({ theme }) => theme.colors.muted};
  font-size: 0.95rem;
`;

const pulseCore = keyframes`
  0% {
    transform: scale(1);
    opacity: 0.85;
  }
  55% {
    transform: scale(1.14);
    opacity: 1;
  }
  100% {
    transform: scale(1);
    opacity: 0.85;
  }
`;

const pulseRing = keyframes`
  0% {
    opacity: 0.32;
    transform: scale(0.95);
    box-shadow: 0 0 0 0 rgba(255, 122, 26, 0.14);
  }
  55% {
    opacity: 0.78;
    transform: scale(1.08);
    box-shadow: 0 0 0 10px rgba(255, 122, 26, 0.1);
  }
  100% {
    opacity: 0;
    transform: scale(1.12);
    box-shadow: 0 0 0 14px rgba(255, 122, 26, 0);
  }
`;

const StatusDot = styled.span<{ $status: AnalysisStatus }>`
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 12px;
  height: 12px;
  border-radius: 999px;
  background: ${({ theme, $status }) => {
    if ($status === 'good') return theme.colors.success;
    if ($status === 'bad') return theme.colors.neon;
    return 'rgba(0,0,0,0.18)';
  }};
  box-shadow: ${({ $status }) => {
    if ($status === 'good') return `0 0 0 4px rgba(28, 168, 106, 0.18)`;
    if ($status === 'bad') return `0 0 0 4px rgba(255, 122, 26, 0.18)`;
    return '0 0 0 2px rgba(0,0,0,0.14)';
  }};
  border: 1px solid rgba(0, 0, 0, 0.05);
  animation: ${({ $status }) => ($status === 'bad' ? css`${pulseCore} 2.8s ease-in-out infinite` : 'none')};
  isolation: isolate;

  &::after {
    content: '';
    position: absolute;
    inset: -4px;
    border-radius: inherit;
    pointer-events: none;
    opacity: 0;
    box-shadow: 0 0 0 0 rgba(255, 122, 26, 0.14);
    transform: scale(0.9);
    animation: ${({ $status }) =>
      $status === 'bad' ? css`${pulseRing} 3.0s ease-in-out infinite` : 'none'};
  }

  @media (prefers-reduced-motion: reduce) {
    &::after {
      animation: none;
    }
    animation: none;
  }
`;

const CriteriaLabel = styled.span`
  color: ${({ theme }) => theme.colors.text};
  font-weight: 600;
`;

const CriteriaWrapper = styled.div`
  display: flex;
  flex-direction: column;
  gap: 6px;
`;

const Actions = styled.div`
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 0;
`;

interface Props {
  profileName: string;
  onUpload: (file: File) => void;
  onViewAnalysis: (analysisId: string) => Promise<void> | void;
  onRegenerateReports: (analysisId: string) => Promise<RegenerateReportsResult> | RegenerateReportsResult;
  onOpenGhostSelector: () => void;
  onOpenCoachChat: () => void;
}

export function DashboardHome({
  profileName,
  onUpload,
  onViewAnalysis,
  onRegenerateReports,
  onOpenGhostSelector,
  onOpenCoachChat,
}: Props) {
  const [analyses, setAnalyses] = useState<AnalysisSummary[]>([]);
  const [pendingRunId, setPendingRunId] = useState<string | null>(null);
  const [modal, setModal] = useState<{ tone: 'success' | 'warn'; message: string } | null>(null);
  const [exclusionOpen, setExclusionOpen] = useState(false);
  const [excludedRuns, setExcludedRuns] = useState<Set<string>>(new Set());
  const [exclusionBusy, setExclusionBusy] = useState(false);
  const criteria: { key: AnalysisCriterion; label: string }[] = [
    { key: 'timing', label: 'Timing' },
    { key: 'spine', label: 'Straight spine' },
    { key: 'drawForce', label: 'Draw-force line' },
    { key: 'drawLength', label: 'Draw length' },
    { key: 'release', label: 'Release' },
    { key: 'followThrough', label: 'Follow-through stability' },
  ];

  useEffect(() => {
    fetchAvailableAnalyses().then(setAnalyses).catch(() => setAnalyses([]));
  }, []);

  const openExclusionManager = async () => {
    setExclusionOpen(true);
    setExclusionBusy(true);
    try {
      const { excluded } = await getRunExclusions();
      setExcludedRuns(new Set(excluded));
    } catch (err) {
      console.error('Failed to load run exclusions', err);
      setExcludedRuns(new Set());
    } finally {
      setExclusionBusy(false);
    }
  };

  const handleSaveExclusions = async () => {
    setExclusionBusy(true);
    try {
      const payload = Array.from(excludedRuns);
      await saveRunExclusions(payload);
      setModal({ tone: 'success', message: 'Run exclusions updated.' });
      setExclusionOpen(false);
    } catch (err) {
      console.error('Failed to save run exclusions', err);
      setModal({ tone: 'warn', message: 'Unable to update run exclusions.' });
    } finally {
      setExclusionBusy(false);
    }
  };

  const handleRegenerate = async (analysisId: string) => {
    setPendingRunId(analysisId);
    try {
      const result = await onRegenerateReports(analysisId);
      const refreshed = await fetchAvailableAnalyses();
      setAnalyses(refreshed);
        setModal({
          tone: result.warnings?.length ? 'warn' : 'success',
          message: result.warnings?.length ? 'Regenerated with some warnings.' : 'Reports Regenerated.',
        });
    } catch (err) {
      console.error('Report regeneration failed', err);
      setModal({ tone: 'warn', message: `Unable to regenerate reports for ${analysisId}.` });
    } finally {
      setPendingRunId((current) => (current === analysisId ? null : current));
    }
  };

  return (
    <Container>
      {modal && (
        <ModalBackdrop role="dialog" aria-modal="true">
          <ModalCard $tone={modal.tone}>
            <ModalMessage>{modal.message}</ModalMessage>
            <ModalActions>
              <ModalButton type="button" onClick={() => setModal(null)}>
                OK
              </ModalButton>
            </ModalActions>
          </ModalCard>
        </ModalBackdrop>
      )}
      {exclusionOpen && (
        <ModalBackdrop role="dialog" aria-modal="true">
          <ExclusionPanel>
            <ExclusionHeader>
              <div>
                <CoachTitle style={{ margin: 0 }}>Exclude runs from averages</CoachTitle>
                <Subtitle style={{ marginTop: 4 }}>Keep outliers from skewing timing and draw-force baselines.</Subtitle>
              </div>
              <ModalButton type="button" onClick={() => setExclusionOpen(false)}>
                Close
              </ModalButton>
            </ExclusionHeader>
            {exclusionBusy && <Subtitle>Loading…</Subtitle>}
            {!exclusionBusy && (
              <ExclusionList>
                {analyses.map((item) => {
                  const checked = excludedRuns.has(item.id);
                  return (
                    <ExclusionItem key={item.id}>
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={(event) => {
                          const next = new Set(excludedRuns);
                          if (event.target.checked) {
                            next.add(item.id);
                          } else {
                            next.delete(item.id);
                          }
                          setExcludedRuns(next);
                        }}
                      />
                      <div>
                        <strong>{item.label}</strong>
                        <div style={{ color: '#8c6f5a', fontSize: '0.9rem' }}>Exclude from dataset averages</div>
                      </div>
                    </ExclusionItem>
                  );
                })}
              </ExclusionList>
            )}
            <ExclusionActions>
              <SecondaryButton type="button" onClick={() => setExclusionOpen(false)} disabled={exclusionBusy}>
                Cancel
              </SecondaryButton>
              <ViewButton type="button" onClick={handleSaveExclusions} disabled={exclusionBusy}>
                {exclusionBusy ? 'Saving…' : 'Save'}
              </ViewButton>
            </ExclusionActions>
          </ExclusionPanel>
        </ModalBackdrop>
      )}
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
          <IconButton as="button" type="button" onClick={openExclusionManager} aria-label="Manage dataset exclusions">
            <IconSvg viewBox="0 0 24 24" role="img" aria-hidden="true">
              <path d="M4 5h16v2H4zm3 6h10v2H7zm4 6h2v2h-2z" />
            </IconSvg>
          </IconButton>
          <IconButton as="button" type="button" onClick={onOpenGhostSelector} aria-label="Choose ghost reference">
            <IconSvg viewBox="0 0 24 24" role="img" aria-hidden="true">
              <path d="M12 2c-3.87 0-7 3.13-7 7v10.5a.5.5 0 0 0 .85.35L7 18.7l1.15 1.15a.5.5 0 0 0 .7 0L10 18.7l1.15 1.15a.5.5 0 0 0 .7 0L13 18.7l1.15 1.15a.5.5 0 0 0 .85-.35V9c0-1.93 1.57-3.5 3.5-3.5S22 7.07 22 9v6.5a.5.5 0 0 0 .85.35L24 14v-5c0-3.87-3.13-7-7-7Zm-4 7.5c-.55 0-1-.45-1-1s.45-1 1-1 1 .45 1 1-.45 1-1 1Zm5 0c-.55 0-1-.45-1-1s.45-1 1-1 1 .45 1 1-.45 1-1 1Z" />
            </IconSvg>
          </IconButton>
          <IconButton href="https://github.com/mont1y/STRAIGHT" target="_blank" rel="noreferrer" aria-label="GitHub repository">
            <IconSvg viewBox="0 0 24 24" role="img" aria-hidden="true">
              <path d="M12 .5C5.65.5.5 5.65.5 12c0 5.09 3.29 9.4 7.86 10.93.58.1.79-.25.79-.56 0-.28-.01-1.02-.02-2-3.2.7-3.88-1.54-3.88-1.54-.53-1.35-1.28-1.7-1.28-1.7-1.05-.72.08-.7.08-.7 1.16.08 1.78 1.2 1.78 1.2 1.03 1.77 2.7 1.26 3.35.96.1-.75.4-1.26.72-1.55-2.55-.29-5.23-1.28-5.23-5.72 0-1.26.45-2.29 1.2-3.1-.12-.29-.52-1.45.12-3.02 0 0 .98-.31 3.2 1.18.93-.26 1.93-.4 2.92-.4.99 0 1.99.14 2.92.4 2.22-1.49 3.2-1.18 3.2-1.18.64 1.57.24 2.73.12 3.02.75.81 1.2 1.84 1.2 3.1 0 4.45-2.69 5.43-5.25 5.71.41.36.77 1.08.77 2.18 0 1.58-.02 2.86-.02 3.25 0 .31.21.67.8.55A10.52 10.52 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5Z" />
            </IconSvg>
          </IconButton>
        </NavActions>
      </HeaderBar>
      <div>
        <Title>Welcome back, {profileName}.</Title>
        <Subtitle>Let’s see if you’ve improved.</Subtitle>
      </div>
      <CoachCard>
        <CoachMeta>
          <CoachBadge src="/coach.png" alt="Coach D logo" />
          <CoachText>
            <CoachTitle>Talk to Coach D.</CoachTitle>
            <CoachSubtitle>Personalized guidance based on your shots, your mistakes, and expert sources.</CoachSubtitle>
          </CoachText>
        </CoachMeta>
        <CoachButton type="button" onClick={onOpenCoachChat}>
          Open coach chat →
        </CoachButton>
      </CoachCard>
      <UploadGrid>
        <UploadNote tabIndex={0}>
          <HiddenInput
            type="file"
            accept="video/mp4,video/quicktime"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) onUpload(file);
            }}
          />
          <UploadButton>Upload a Shot →</UploadButton>
          <Subtitle>Drop your latest rep or click to browse files. MP4/MOV under 2GB works best.</Subtitle>
        </UploadNote>
        <TipsNote>
          <ExampleFrame>
            <ExampleImage src="/example.png" alt="Example recording setup" />
          </ExampleFrame>
          <TipBody>
            <TipTitle>Record it right</TipTitle>
            <StepList>
              <StepItem>
                <StepNum>1</StepNum>
                <StepText>Place the camera directly from your side (90° to the shooting line).</StepText>
              </StepItem>
              <StepItem>
                <StepNum>2</StepNum>
                <StepText>Ensure your entire body is fully visible in the frame.</StepText>
              </StepItem>
              <StepItem>
                <StepNum>3</StepNum>
                <StepText>Keep the camera angle consistent for every shot.</StepText>
              </StepItem>
            </StepList>
          </TipBody>
        </TipsNote>
      </UploadGrid>
      <AnalysesSection aria-label="Available analyses">
        <AnalysesHeading>Available analyses</AnalysesHeading>
        {analyses.length ? (
          <AnalysesList>
            {analyses.map((item) => (
              <AnalysisItem key={item.id}>
                <TitleRow>
                  <CriteriaWrapper>
                    <AnalysisLabel>{item.label}</AnalysisLabel>
                    <Subtitle style={{ marginTop: 2 }}>Quick pulse across your key checkpoints.</Subtitle>
                  </CriteriaWrapper>
                  <Actions>
                    <SecondaryButton
                      type="button"
                      disabled={pendingRunId === item.id}
                      onClick={() => handleRegenerate(item.id)}
                    >
                      {pendingRunId === item.id ? 'Running…' : 'Regenerate Reports'}
                    </SecondaryButton>
                    <ViewButton type="button" disabled={pendingRunId === item.id} onClick={() => onViewAnalysis(item.id)}>
                      View Full Report
                    </ViewButton>
                  </Actions>
                </TitleRow>
                <CriteriaGrid>
                  {criteria.map((crit) => {
                    const status = item.overview?.[crit.key] ?? 'unknown';
                    return (
                      <CriteriaItem key={crit.key}>
                        <StatusDot $status={status as AnalysisStatus} aria-label={status} />
                        <CriteriaLabel>{crit.label}</CriteriaLabel>
                      </CriteriaItem>
                    );
                  })}
                </CriteriaGrid>
              </AnalysisItem>
            ))}
          </AnalysesList>
        ) : (
          <Subtitle>No analyses found in assets.</Subtitle>
        )}
      </AnalysesSection>
    </Container>
  );
}
