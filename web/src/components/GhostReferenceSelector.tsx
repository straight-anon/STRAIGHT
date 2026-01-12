import { useEffect, useState } from 'react';
import styled from 'styled-components';
import logoBadge from '../assets/tpn_logo.png';
import { fetchAvailableAnalyses, getGhostReference, setGhostReference, type AnalysisSummary } from '../services/api';

type Props = {
  onBack: () => void;
};

export function GhostReferenceSelector({ onBack }: Props) {
  const [analyses, setAnalyses] = useState<AnalysisSummary[]>([]);
  const [currentRef, setCurrentRef] = useState<string | null>(null);
  const [savingId, setSavingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchAvailableAnalyses().then(setAnalyses).catch(() => setAnalyses([]));
    getGhostReference()
      .then((resp) => setCurrentRef(resp.referenceId))
      .catch(() => setCurrentRef(null));
  }, []);

  const handleSelect = async (id: string) => {
    setSavingId(id);
    setError(null);
    try {
      const resp = await setGhostReference(id);
      setCurrentRef(resp.referenceId);
    } catch (err) {
      console.error(err);
      setError('Unable to update ghost reference.');
    } finally {
      setSavingId(null);
    }
  };

  return (
    <Container>
      <HeaderBar>
        <BrandGroup>
          <LogoBadge src={logoBadge} alt="TPN logo" />
          <BrandText>Ghost Reference</BrandText>
        </BrandGroup>
        <NavActions aria-label="Ghost reference navigation">
          <IconButton type="button" onClick={onBack} aria-label="Return to dashboard">
            ←
          </IconButton>
        </NavActions>
      </HeaderBar>
      <Copy>
        <Title>Pick the shot that should drive your ghost overlay.</Title>
        <Subtitle>
          We’ll reuse this reference for new analyses and re-run timing comparisons when dataset stats are available.
        </Subtitle>
        {error && <ErrorBanner role="status">{error}</ErrorBanner>}
      </Copy>
      <CardGrid>
        {analyses.map((run) => (
          <Card key={run.id} aria-label={`Ghost reference option ${run.label}`}>
            <CardHeader>
              <CardTitle>{run.label}</CardTitle>
              {currentRef === run.id && <Pill>Current</Pill>}
            </CardHeader>
            <CardSubtitle>ID: {run.id}</CardSubtitle>
            <SelectButton type="button" disabled={savingId === run.id} onClick={() => handleSelect(run.id)}>
              {savingId === run.id ? 'Saving…' : 'Use as ghost'}
            </SelectButton>
          </Card>
        ))}
        {!analyses.length && <EmptyState>No saved analyses found.</EmptyState>}
      </CardGrid>
    </Container>
  );
}

const Container = styled.section`
  min-height: 100vh;
  padding: ${({ theme }) => theme.spacing(4)};
  background: radial-gradient(circle at top left, rgba(140, 103, 255, 0.15), transparent 40%),
    ${({ theme }) => theme.colors.background};
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing(3)};
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

const LogoBadge = styled.img`
  height: 38px;
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

const IconButton = styled.button`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 40px;
  padding: 0 ${({ theme }) => theme.spacing(1.5)};
  border-radius: 10px;
  border: 1px solid rgba(0, 0, 0, 0.08);
  background: ${({ theme }) => theme.colors.surfaceAlt};
  color: ${({ theme }) => theme.colors.text};
  cursor: pointer;
  font-weight: 700;
  letter-spacing: 0.01em;
  transition: transform ${({ theme }) => theme.durations.quick} ease, box-shadow ${({ theme }) =>
      theme.durations.quick} ease;

  &:hover,
  &:focus-visible {
    outline: none;
    transform: translateY(-1px);
    box-shadow: ${({ theme }) => theme.shadows.glow};
  }
`;

const Copy = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing(0.5)};
  max-width: 840px;
`;

const Title = styled.h2`
  margin: 0;
  font-family: ${({ theme }) => theme.fonts.heading};
`;

const Subtitle = styled.p`
  margin: 0;
  color: ${({ theme }) => theme.colors.muted};
`;

const CardGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: ${({ theme }) => theme.spacing(1.5)};
`;

const Card = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing(1)};
  padding: ${({ theme }) => theme.spacing(2)};
  border-radius: ${({ theme }) => theme.radii.lg};
  background: ${({ theme }) => theme.colors.surface};
  border: 1px solid rgba(0, 0, 0, 0.04);
  box-shadow: ${({ theme }) => theme.shadows.glow};
`;

const CardHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: ${({ theme }) => theme.spacing(1)};
`;

const CardTitle = styled.h3`
  margin: 0;
  font-size: 1.05rem;
`;

const CardSubtitle = styled.span`
  color: ${({ theme }) => theme.colors.muted};
`;

const SelectButton = styled.button`
  align-self: flex-start;
  padding: 8px 14px;
  border-radius: 10px;
  border: 1px solid rgba(255, 122, 26, 0.24);
  background: linear-gradient(120deg, rgba(255, 245, 234, 0.95), rgba(255, 224, 192, 0.9));
  cursor: pointer;
  font-weight: 700;
  letter-spacing: 0.01em;
  transition: transform ${({ theme }) => theme.durations.quick} ease, box-shadow ${({ theme }) =>
      theme.durations.quick} ease;

  &:hover,
  &:focus-visible {
    outline: none;
    transform: translateY(-1px);
    box-shadow: ${({ theme }) => theme.shadows.glow};
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }
`;

const Pill = styled.span`
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(70, 248, 194, 0.16);
  color: ${({ theme }) => theme.colors.text};
  border: 1px solid rgba(70, 248, 194, 0.4);
  font-weight: 700;
`;

const ErrorBanner = styled.div`
  padding: ${({ theme }) => theme.spacing(1.25)};
  border-radius: ${({ theme }) => theme.radii.md};
  background: rgba(255, 92, 88, 0.14);
  border: 1px solid rgba(255, 92, 88, 0.38);
  color: #641414;
  font-weight: 700;
`;

const EmptyState = styled.div`
  padding: ${({ theme }) => theme.spacing(2)};
  border-radius: ${({ theme }) => theme.radii.md};
  background: ${({ theme }) => theme.colors.surfaceAlt};
  color: ${({ theme }) => theme.colors.muted};
  border: 1px dashed rgba(0, 0, 0, 0.08);
`;
