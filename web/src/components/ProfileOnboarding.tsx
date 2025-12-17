import { useState } from 'react';
import styled, { keyframes } from 'styled-components';
import type { Profile } from '../services/api';
import logoBadge from '../assets/tpn_logo.png';

const fadeIn = keyframes`
  from { opacity: 0; transform: translateY(16px); }
  to { opacity: 1; transform: translateY(0); }
`;

const Shell = styled.section`
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: ${({ theme }) => theme.spacing(4)};
  background: radial-gradient(circle at top, rgba(70, 248, 194, 0.08), transparent 55%);
`;

const Panel = styled.div`
  width: min(560px, 100%);
  background: ${({ theme }) => theme.colors.surface};
  border-radius: ${({ theme }) => theme.radii.lg};
  padding: ${({ theme }) => theme.spacing(5)};
  box-shadow: ${({ theme }) => theme.shadows.glow};
  animation: ${fadeIn} ${({ theme }) => theme.durations.normal} ease;
`;

const Heading = styled.h1`
  margin: 0 0 ${({ theme }) => theme.spacing(2)};
  font-family: ${({ theme }) => theme.fonts.heading};
  letter-spacing: 0.04em;
`;

const Description = styled.p`
  margin: 0 0 ${({ theme }) => theme.spacing(4)};
  color: ${({ theme }) => theme.colors.muted};
`;

const PrimaryButton = styled.button`
  width: 100%;
  padding: ${({ theme }) => theme.spacing(2)};
  border: none;
  border-radius: ${({ theme }) => theme.radii.md};
  background: linear-gradient(120deg, ${({ theme }) => theme.colors.neon}, ${({ theme }) => theme.colors.neonAlt});
  color: #05070f;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: filter ${({ theme }) => theme.durations.quick} ease;

  &:hover,
  &:focus-visible {
    filter: brightness(1.1);
    outline: none;
  }
`;

const Input = styled.input`
  width: 100%;
  padding: ${({ theme }) => theme.spacing(2)};
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: ${({ theme }) => theme.colors.surfaceAlt};
  color: ${({ theme }) => theme.colors.text};
  font-size: 1.1rem;
  transition: border-color ${({ theme }) => theme.durations.quick} ease;

  &:focus-visible {
    outline: none;
    border-color: ${({ theme }) => theme.colors.neon};
    box-shadow: 0 0 0 2px rgba(70, 248, 194, 0.25);
  }
`;

const Textarea = styled.textarea`
  width: 100%;
  min-height: 120px;
  padding: ${({ theme }) => theme.spacing(2)};
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: ${({ theme }) => theme.colors.surfaceAlt};
  color: ${({ theme }) => theme.colors.text};
  font-size: 1rem;
  resize: vertical;
  transition: border-color ${({ theme }) => theme.durations.quick} ease;

  &:focus-visible {
    outline: none;
    border-color: ${({ theme }) => theme.colors.neon};
    box-shadow: 0 0 0 2px rgba(70, 248, 194, 0.25);
  }
`;

const StepIndicator = styled.div`
  display: flex;
  gap: ${({ theme }) => theme.spacing(0.5)};
  margin-bottom: ${({ theme }) => theme.spacing(4)};
  align-items: center;
`;

const LogoMark = styled.img`
  height: 64px;
  width: auto;
`;

const FieldLabel = styled.label`
  display: block;
  margin: ${({ theme }) => theme.spacing(3)} 0 ${({ theme }) => theme.spacing(1)};
  font-weight: 600;
`;

const HelperText = styled.p`
  margin: 0 0 ${({ theme }) => theme.spacing(1.5)};
  color: ${({ theme }) => theme.colors.muted};
  font-size: 0.95rem;
`;

type Step = 'intro' | 'name' | 'draw';

interface Props {
  onComplete: (profile: Profile) => Promise<void> | void;
}

export function ProfileOnboarding({ onComplete }: Props) {
  const [step, setStep] = useState<Step>('intro');
  const [name, setName] = useState('');
  const [drawWeight, setDrawWeight] = useState('');
  const [notes, setNotes] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async () => {
    if (!name.trim() || !drawWeight.trim()) return;
    setSubmitting(true);
    await onComplete({ name: name.trim(), drawWeight: drawWeight.trim(), notes: notes.trim() });
    setSubmitting(false);
  };

  return (
    <Shell>
      <Panel>
        <StepIndicator aria-hidden>
          <LogoMark src={logoBadge} alt="TPN logo" />
        </StepIndicator>
        {step === 'intro' && (
          <>
            <Heading>Ready to start shooting?</Heading>
            <Description>Let’s set up your archer profile so every analysis is tailored to you.</Description>
            <PrimaryButton onClick={() => setStep('name')}>Create profile</PrimaryButton>
          </>
        )}

        {step === 'name' && (
          <form
            onSubmit={(event) => {
              event.preventDefault();
              if (name.trim()) {
                setStep('draw');
              }
            }}
          >
            <Heading>What’s your name?</Heading>
            <Description>We’ll greet you properly every session.</Description>
            <Input
              aria-label="Enter your name"
              value={name}
              onChange={(event) => setName(event.target.value)}
              autoFocus
            />
            <PrimaryButton style={{ marginTop: '24px' }} type="submit">
              Continue
            </PrimaryButton>
          </form>
        )}

        {step === 'draw' && (
          <form
            onSubmit={(event) => {
              event.preventDefault();
              handleSubmit();
            }}
          >
            <Heading>What’s your draw weight?</Heading>
            <Description>Numbers help us calibrate recommendations.</Description>
            <Input
              aria-label="Enter your draw weight"
              value={drawWeight}
              onChange={(event) => setDrawWeight(event.target.value)}
              autoFocus
            />
            <FieldLabel htmlFor="additional-notes">
              What additional notes would you like the coach to know?
            </FieldLabel>
            <HelperText>Medical conditions, injuries, equipment quirks, or anything else we should consider.</HelperText>
            <Textarea
              id="additional-notes"
              aria-label="Add any notes for your coach"
              placeholder="Share relevant details here"
              value={notes}
              onChange={(event) => setNotes(event.target.value)}
            />
            <PrimaryButton
              disabled={submitting || !drawWeight.trim()}
              style={{ marginTop: '24px' }}
              type="submit"
            >
              {submitting ? 'Saving…' : 'Save profile'}
            </PrimaryButton>
          </form>
        )}
      </Panel>
    </Shell>
  );
}
