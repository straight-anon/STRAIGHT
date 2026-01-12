import styled from 'styled-components';

interface Props {
  filename?: string;
  onSelect: (useShotTrainer: boolean) => void;
  onCancel: () => void;
}

export function ModeSelect({ filename, onSelect, onCancel }: Props) {
  return (
    <Container>
      <Panel>
        <Header>
          <div>
            <Eyebrow>Choose your setup</Eyebrow>
            <Title>How are you shooting?</Title>
            <Subtitle>Select the mode that matches this upload to calibrate the analysis.</Subtitle>
            {filename && <Filename>Selected file: {filename}</Filename>}
          </div>
          <BackButton type="button" onClick={onCancel}>
            ← Back
          </BackButton>
        </Header>
        <CardGrid>
          <ModeCard type="button" onClick={() => onSelect(false)}>
            <Badge>Standard bow</Badge>
            <CardTitle>Bow only</CardTitle>
            <CardBody>
              <li>Detects draw, release, and follow-through without trainer cues.</li>
              <li>Use for barebow or non-instrumented sessions.</li>
            </CardBody>
          </ModeCard>
          <ModeCard type="button" onClick={() => onSelect(true)}>
            <Badge $accent>Shot trainer</Badge>
            <CardTitle>With shot trainer</CardTitle>
            <CardBody>
              <li>Optimizes prompts for trainer-assisted timing.</li>
              <li>Use when a shot trainer is visible or active.</li>
            </CardBody>
          </ModeCard>
        </CardGrid>
      </Panel>
    </Container>
  );
}

const Container = styled.section`
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: ${({ theme }) => theme.spacing(5)};
  background: radial-gradient(circle at 16% 18%, rgba(255, 122, 26, 0.12), transparent 32%),
    radial-gradient(circle at 82% 12%, rgba(140, 103, 255, 0.14), transparent 36%),
    ${({ theme }) => theme.colors.background};
`;

const Panel = styled.div`
  width: min(900px, 100%);
  background: ${({ theme }) => theme.colors.surface};
  border-radius: ${({ theme }) => theme.radii.lg};
  padding: ${({ theme }) => theme.spacing(4)};
  box-shadow: ${({ theme }) => theme.shadows.glow};
  border: 1px solid rgba(0, 0, 0, 0.04);
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing(3)};
`;

const Header = styled.div`
  display: flex;
  justify-content: space-between;
  gap: ${({ theme }) => theme.spacing(2)};
  align-items: flex-start;

  @media (max-width: 720px) {
    flex-direction: column;
  }
`;

const Eyebrow = styled.span`
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: ${({ theme }) => theme.colors.neonAlt};
  font-weight: 700;
`;

const Title = styled.h1`
  margin: 6px 0 6px;
  font-family: ${({ theme }) => theme.fonts.heading};
  color: ${({ theme }) => theme.colors.text};
`;

const Subtitle = styled.p`
  margin: 0;
  color: ${({ theme }) => theme.colors.muted};
`;

const Filename = styled.p`
  margin: ${({ theme }) => theme.spacing(1)} 0 0;
  color: ${({ theme }) => theme.colors.text};
  font-weight: 600;
`;

const BackButton = styled.button`
  align-self: flex-start;
  border: 1px solid rgba(0, 0, 0, 0.08);
  background: ${({ theme }) => theme.colors.surfaceAlt};
  color: ${({ theme }) => theme.colors.text};
  padding: 10px 14px;
  border-radius: ${({ theme }) => theme.radii.md};
  cursor: pointer;
  font-weight: 700;
  transition: transform ${({ theme }) => theme.durations.quick} ease, box-shadow ${({ theme }) => theme.durations.quick} ease;

  &:hover,
  &:focus-visible {
    transform: translateY(-1px);
    box-shadow: ${({ theme }) => theme.shadows.glow};
    outline: none;
  }
`;

const CardGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: ${({ theme }) => theme.spacing(2)};
`;

const ModeCard = styled.button`
  position: relative;
  text-align: left;
  padding: ${({ theme }) => theme.spacing(3)};
  border-radius: ${({ theme }) => theme.radii.lg};
  border: 1px solid rgba(0, 0, 0, 0.06);
  background: linear-gradient(140deg, rgba(255, 255, 255, 0.96), rgba(255, 241, 224, 0.92));
  box-shadow: ${({ theme }) => theme.shadows.glow};
  cursor: pointer;
  transition: transform ${({ theme }) => theme.durations.quick} ease, box-shadow ${({ theme }) => theme.durations.quick} ease,
    border-color ${({ theme }) => theme.durations.quick} ease;

  &:hover,
  &:focus-visible {
    transform: translateY(-3px);
    box-shadow: ${({ theme }) => theme.shadows.glow}, 0 14px 40px rgba(0, 0, 0, 0.08);
    border-color: rgba(255, 122, 26, 0.38);
    outline: none;
  }
`;

const Badge = styled.span<{ $accent?: boolean }>`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 6px 10px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 0.8rem;
  color: #05070f;
  background: ${({ $accent, theme }) => ($accent ? theme.colors.neon : theme.colors.surfaceAlt)};
  border: 1px solid rgba(0, 0, 0, 0.06);
`;

const CardTitle = styled.h3`
  margin: ${({ theme }) => theme.spacing(1)} 0 ${({ theme }) => theme.spacing(1.5)};
  color: ${({ theme }) => theme.colors.text};
`;

const CardBody = styled.ul`
  margin: 0;
  padding-left: ${({ theme }) => theme.spacing(2.5)};
  color: ${({ theme }) => theme.colors.muted};
  display: grid;
  gap: ${({ theme }) => theme.spacing(0.75)};
`;
