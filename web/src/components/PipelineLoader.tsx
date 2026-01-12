import styled, { keyframes } from 'styled-components';
import type { PipelineProgress } from '../services/api';

const spin = keyframes`
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
`;

const Shell = styled.section`
  min-height: 100vh;
  display: grid;
  place-items: center;
  padding: ${({ theme }) => theme.spacing(6)};
  background:
    radial-gradient(circle at 14% 22%, rgba(255, 122, 26, 0.12), transparent 32%),
    radial-gradient(circle at 82% 12%, rgba(140, 103, 255, 0.14), transparent 36%),
    ${({ theme }) => theme.colors.background};
`;

const Panel = styled.div`
  width: min(760px, 100%);
  background: ${({ theme }) => theme.colors.surface};
  border-radius: ${({ theme }) => theme.radii.lg};
  padding: ${({ theme }) => theme.spacing(5)};
  box-shadow: ${({ theme }) => theme.shadows.glow};
  display: grid;
  grid-template-columns: 1fr;
  gap: ${({ theme }) => theme.spacing(3)};
`;

const StageList = styled.ol`
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: ${({ theme }) => theme.spacing(1.25)};
`;

const StageItem = styled.li<{ $status: PipelineProgress['status'] }>`
  display: grid;
  grid-template-columns: 32px 1fr;
  align-items: center;
  gap: ${({ theme }) => theme.spacing(1)};
  padding: ${({ theme }) => theme.spacing(1.25)} ${({ theme }) => theme.spacing(1.5)};
  border-radius: ${({ theme }) => theme.radii.md};
  background: ${({ theme }) => theme.colors.surfaceAlt};
  border: 1px solid rgba(0, 0, 0, 0.06);
  box-shadow: ${({ theme }) => theme.shadows.glow};
  position: relative;
  overflow: hidden;
  isolation: isolate;
`;

const StageDot = styled.span<{ $status: PipelineProgress['status'] }>`
  width: 18px;
  height: 18px;
  border-radius: 50%;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: ${({ $status, theme }) => ($status === 'done' ? theme.colors.neon : 'rgba(0, 0, 0, 0.25)')};
  box-shadow: ${({ $status }) => ($status === 'done' ? `0 0 0 6px rgba(255, 122, 26, 0.16)` : 'none')};
  position: relative;
  z-index: 1;
`;

const ArrowCard = styled.div`
  position: relative;
  overflow: hidden;
  border-radius: ${({ theme }) => theme.radii.lg};
  background: linear-gradient(145deg, rgba(255, 248, 240, 0.96), rgba(255, 241, 224, 0.92));
  padding: ${({ theme }) => theme.spacing(3)};
  color: ${({ theme }) => theme.colors.text};
  box-shadow: ${({ theme }) => theme.shadows.glow};
  display: grid;
  gap: ${({ theme }) => theme.spacing(1.25)};
  isolation: isolate;
`;

const ArrowSpinner = styled.div`
  width: 58px;
  height: 58px;
  border-radius: 50%;
  position: relative;
  display: grid;
  place-items: center;
  background: radial-gradient(circle at 30% 30%, rgba(255, 255, 255, 0.9), rgba(255, 122, 26, 0.12));
  box-shadow: 0 12px 28px rgba(0, 0, 0, 0.08), 0 0 0 8px rgba(255, 122, 26, 0.1);

  &::before {
    content: '';
    position: absolute;
    inset: 8px;
    border-radius: 50%;
    background:
      conic-gradient(from 0deg, rgba(255, 122, 26, 0.45), rgba(255, 122, 26, 0), rgba(255, 122, 26, 0));
    mask: radial-gradient(circle at center, black 60%, transparent 61%);
    animation: ${spin} 1.8s linear infinite;
  }

  &::after {
    content: '';
    position: absolute;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: ${({ theme }) => theme.colors.neon};
    box-shadow: 0 0 12px rgba(255, 122, 26, 0.35);
  }
`;

const ArrowTitle = styled.div`
  position: relative;
  z-index: 1;
  font-family: ${({ theme }) => theme.fonts.heading};
  font-size: 1.2rem;
  color: ${({ theme }) => theme.colors.text};
`;

const ArrowStatus = styled.div`
  position: relative;
  z-index: 1;
  color: ${({ theme }) => theme.colors.muted};
`;

const StatusLabel = styled.p`
  margin-top: ${({ theme }) => theme.spacing(2)};
  color: ${({ theme }) => theme.colors.muted};
`;

interface Props {
  steps: PipelineProgress[];
  tipText?: string;
}

export function PipelineLoader({ steps, tipText }: Props) {
  const activeStep = steps.find((s) => s.status === 'active') ?? steps.find((s) => s.status === 'pending') ?? steps[0];
  return (
    <Shell aria-live="polite">
      <Panel>
        <ArrowCard>
          <ArrowTitle>Processing your run…</ArrowTitle>
          <ArrowStatus>{activeStep?.label ?? 'Working…'}</ArrowStatus>
          <ArrowSpinner aria-hidden />
        </ArrowCard>
        <StageList>
          {steps.map((step) => (
            <StageItem key={step.id} $status={step.status}>
              <StageDot $status={step.status} />
              {step.label}
            </StageItem>
          ))}
        </StageList>
        {tipText && <StatusLabel>{tipText}</StatusLabel>}
      </Panel>
    </Shell>
  );
}
