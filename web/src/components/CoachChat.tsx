import { useEffect, useMemo, useRef, useState, type KeyboardEventHandler } from 'react';
import styled from 'styled-components';
import { fetchAvailableAnalyses, sendCoachChat, type AnalysisSummary, type CoachChatMessage } from '../services/api';

type Message = { id: string; role: 'coach' | 'user'; text: string; pending?: boolean };

type Props = {
  profileName?: string | null;
  onBack: () => void;
};

const Page = styled.div`
  min-height: 100vh;
  background: radial-gradient(circle at 20% 20%, rgba(255, 122, 26, 0.12), transparent 32%),
    radial-gradient(circle at 70% 10%, rgba(140, 103, 255, 0.12), transparent 30%),
    ${({ theme }) => theme.colors.background};
  padding: ${({ theme }) => theme.spacing(5)} ${({ theme }) => theme.spacing(6)};
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing(3)};
`;

const Header = styled.header`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: ${({ theme }) => theme.spacing(2)};
`;

const Brand = styled.div`
  display: inline-flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing(1.25)};
  padding: ${({ theme }) => theme.spacing(1.25)} ${({ theme }) => theme.spacing(1.5)};
  border-radius: ${({ theme }) => theme.radii.md};
  background: ${({ theme }) => theme.colors.surface};
  box-shadow: ${({ theme }) => theme.shadows.glow};
`;

const BrandMark = styled.img`
  width: 42px;
  height: 42px;
  border-radius: ${({ theme }) => theme.radii.md};
  object-fit: cover;
`;

const BrandText = styled.div`
  display: flex;
  flex-direction: column;
  line-height: 1.2;
`;

const BrandTitle = styled.span`
  font-family: ${({ theme }) => theme.fonts.heading};
  font-weight: 700;
  letter-spacing: 0.01em;
`;

const BrandSubtitle = styled.span`
  font-size: 0.9rem;
  color: ${({ theme }) => theme.colors.muted};
`;

const HeaderActions = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing(1)};
  justify-content: flex-end;
`;

const HomeButton = styled.button`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 42px;
  height: 42px;
  border-radius: 12px;
  background: ${({ theme }) => theme.colors.surface};
  color: ${({ theme }) => theme.colors.text};
  border: 1px solid rgba(0, 0, 0, 0.08);
  cursor: pointer;
  transition: transform ${({ theme }) => theme.durations.quick} ease, box-shadow ${({ theme }) => theme.durations.quick} ease;

  &:hover,
  &:focus-visible {
    transform: translateY(-2px);
    box-shadow: ${({ theme }) => theme.shadows.glow};
    outline: none;
  }
`;

const ToggleButton = styled.button`
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: ${({ theme }) => theme.radii.md};
  background: ${({ theme }) => theme.colors.surface};
  color: ${({ theme }) => theme.colors.text};
  padding: ${({ theme }) => theme.spacing(1)} ${({ theme }) => theme.spacing(1.5)};
  font-weight: 700;
  cursor: pointer;
  transition: transform ${({ theme }) => theme.durations.quick} ease, box-shadow ${({ theme }) => theme.durations.quick} ease;

  &:hover,
  &:focus-visible {
    transform: translateY(-1px);
    box-shadow: ${({ theme }) => theme.shadows.glow};
    outline: none;
  }
`;

const Layout = styled.div`
  display: grid;
  grid-template-columns: minmax(320px, 380px) 1fr;
  gap: ${({ theme }) => theme.spacing(2.5)};

  @media (max-width: 980px) {
    grid-template-columns: 1fr;
  }
`;

const Card = styled.div`
  background: ${({ theme }) => theme.colors.surface};
  border-radius: ${({ theme }) => theme.radii.lg};
  border: 1px solid rgba(0, 0, 0, 0.04);
  box-shadow: ${({ theme }) => theme.shadows.glow};
  padding: ${({ theme }) => theme.spacing(2.5)};
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing(1.5)};
`;

const CoachHero = styled(Card)`
  position: relative;
  overflow: hidden;
  padding: ${({ theme }) => theme.spacing(2.5)};
  background: linear-gradient(135deg, rgba(255, 218, 200, 0.9), rgba(255, 244, 232, 0.95));
`;

const CoachName = styled.h2`
  margin: 0;
  font-family: ${({ theme }) => theme.fonts.heading};
  font-size: 1.8rem;
  color: ${({ theme }) => theme.colors.text};
`;

const CoachLine = styled.p`
  margin: 0;
  color: ${({ theme }) => theme.colors.text};
  max-width: 28ch;
`;

const Badge = styled.span`
  display: inline-flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing(0.75)};
  padding: ${({ theme }) => theme.spacing(0.75)} ${({ theme }) => theme.spacing(1.25)};
  background: rgba(0, 0, 0, 0.06);
  border-radius: 999px;
  font-weight: 700;
  color: ${({ theme }) => theme.colors.text};
`;

const Glow = styled.div`
  position: absolute;
  inset: 0;
  background: radial-gradient(circle at 20% 30%, rgba(255, 122, 26, 0.22), transparent 40%),
    radial-gradient(circle at 80% 10%, rgba(140, 103, 255, 0.2), transparent 42%);
  pointer-events: none;
`;

const ChatCard = styled(Card)`
  min-height: 560px;
  background: ${({ theme }) => theme.colors.surface};
  position: relative;
  overflow: hidden;
`;

const ChatHeader = styled.div`
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: ${({ theme }) => theme.spacing(1)};
`;

const ChatTitle = styled.h3`
  margin: 0;
  font-family: ${({ theme }) => theme.fonts.heading};
`;

const ContextHint = styled.span`
  color: ${({ theme }) => theme.colors.muted};
  font-size: 0.95rem;
`;

const RunSelectRow = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing(1)};
  flex-wrap: wrap;
  margin-top: ${({ theme }) => theme.spacing(1)};
`;

const RunSelect = styled.select`
  padding: ${({ theme }) => theme.spacing(1)};
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid rgba(0, 0, 0, 0.1);
  background: ${({ theme }) => theme.colors.surface};
  color: ${({ theme }) => theme.colors.text};
  font-weight: 600;
  box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.05);
`;

const ChatStream = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing(1)};
  padding: ${({ theme }) => theme.spacing(1.5)};
  border-radius: ${({ theme }) => theme.radii.md};
  background: ${({ theme }) => theme.colors.surfaceAlt};
  overflow-y: auto;
  max-height: 640px;
`;

const Bubble = styled.div<{ $role: 'coach' | 'user'; $pending?: boolean }>`
  align-self: ${({ $role }) => ($role === 'user' ? 'flex-end' : 'flex-start')};
  background: ${({ theme, $role }) =>
    $role === 'user'
      ? `linear-gradient(135deg, rgba(255, 218, 200, 0.95), ${theme.colors.neon})`
      : `linear-gradient(135deg, ${theme.colors.surfaceAlt}, rgba(140, 103, 255, 0.12))`};
  color: ${({ theme }) => theme.colors.text};
  padding: ${({ theme }) => theme.spacing(1.25)} ${({ theme }) => theme.spacing(1.5)};
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid rgba(0, 0, 0, 0.05);
  box-shadow: ${({ theme }) => theme.shadows.glow};
  max-width: 82%;
  opacity: ${({ $pending }) => ($pending ? 0.7 : 1)};
`;

const InputArea = styled.div`
  display: grid;
  grid-template-columns: 1fr auto;
  align-items: end;
  gap: ${({ theme }) => theme.spacing(1)};
  padding-top: ${({ theme }) => theme.spacing(1)};
`;

const PromptField = styled.textarea`
  width: 100%;
  min-height: 96px;
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid rgba(0, 0, 0, 0.08);
  padding: ${({ theme }) => theme.spacing(1.25)};
  font-size: 1rem;
  resize: vertical;
  background: ${({ theme }) => theme.colors.surface};
  color: ${({ theme }) => theme.colors.text};
  box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.05);
`;

const SendButton = styled.button<{ $busy?: boolean }>`
  align-self: stretch;
  padding: ${({ theme }) => theme.spacing(1.25)} ${({ theme }) => theme.spacing(1.75)};
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid ${({ theme }) => theme.colors.neonAlt};
  background: linear-gradient(135deg, ${({ theme }) => theme.colors.neon} 0%, rgba(255, 196, 150, 0.96) 100%);
  color: #120c06;
  font-weight: 800;
  cursor: ${({ $busy }) => ($busy ? 'wait' : 'pointer')};
  opacity: ${({ $busy }) => ($busy ? 0.8 : 1)};
  pointer-events: ${({ $busy }) => ($busy ? 'none' : 'auto')};
  box-shadow: ${({ theme }) => theme.shadows.glow};
  min-width: 140px;
  transition: transform ${({ theme }) => theme.durations.quick} ease, box-shadow ${({ theme }) => theme.durations.quick} ease;

  &:hover,
  &:focus-visible {
    transform: ${({ $busy }) => ($busy ? 'none' : 'translateY(-2px)')};
    box-shadow: ${({ theme }) => theme.shadows.glow}, 0 16px 38px rgba(0, 0, 0, 0.08);
    outline: none;
  }
`;

const ErrorText = styled.p`
  margin: 0;
  color: ${({ theme }) => theme.colors.danger};
  font-weight: 600;
`;

const ContextCard = styled(Card)`
  background: ${({ theme }) => theme.colors.surfaceAlt};
`;

const ContextTitle = styled.h4`
  margin: 0;
  font-family: ${({ theme }) => theme.fonts.heading};
`;

const ContextRow = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing(0.5)};
`;

const ContextLabel = styled.span`
  color: ${({ theme }) => theme.colors.muted};
  font-weight: 600;
  letter-spacing: 0.01em;
`;

const ContextBox = styled.div`
  background: ${({ theme }) => theme.colors.surface};
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid rgba(0, 0, 0, 0.04);
  padding: ${({ theme }) => theme.spacing(1.25)};
  min-height: 120px;
  white-space: pre-line;
  color: ${({ theme }) => theme.colors.text};
  box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.05);
`;

const FooterNote = styled.p`
  margin: 0;
  color: ${({ theme }) => theme.colors.muted};
`;

const buildHistoryPayload = (messages: Message[]): CoachChatMessage[] => {
  return messages.filter((msg) => !msg.pending).map((msg) => ({
    role: msg.role === 'coach' ? 'assistant' : 'user',
    content: msg.text,
  }));
};

const welcomeLine = (name?: string | null) =>
  name && name.trim().length
    ? `${name}, I'm Coach D. Give me a shot to review or ask me what's off.`
    : "I'm Coach D. Drop your question and I'll tell you what to fix.";

export function CoachChat({ profileName, onBack }: Props) {
  const [messages, setMessages] = useState<Message[]>([
    { id: 'coach-welcome', role: 'coach', text: welcomeLine(profileName) },
  ]);
  const [input, setInput] = useState('');
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [latestRunId, setLatestRunId] = useState<string | null>(null);
  const [contextPreview, setContextPreview] = useState<string | null>(null);
  const [showContext, setShowContext] = useState(false);
  const [availableRuns, setAvailableRuns] = useState<AnalysisSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [selectedRunLabel, setSelectedRunLabel] = useState<string>('Latest run');
  const streamRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (streamRef.current) {
      streamRef.current.scrollTop = streamRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    fetchAvailableAnalyses()
      .then((list) => setAvailableRuns(list))
      .catch(() => setAvailableRuns([]));
  }, []);

  useEffect(() => {
    // Reset to latest when selection cleared
    if (!selectedRunId) {
      setSelectedRunLabel('Latest run');
    }
  }, [selectedRunId]);

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed || pending) return;

    const userMessage: Message = { id: `user-${Date.now()}`, role: 'user', text: trimmed };
    const pendingId = `pending-${Date.now()}`;
    const pendingMessage: Message = { id: pendingId, role: 'coach', text: 'Thinking...', pending: true };
    const historyPayload = buildHistoryPayload(messages);

    setMessages((prev) => [...prev, userMessage, pendingMessage]);
    setInput('');
    setError(null);
    setPending(true);

    try {
      const response = await sendCoachChat(trimmed, historyPayload, selectedRunId || undefined);
      setMessages((prev) => [
        ...prev.filter((msg) => msg.id !== pendingId),
        { id: `coach-${Date.now()}`, role: 'coach', text: response.reply },
      ]);
      setLatestRunId(response.latestRunId ?? selectedRunId ?? null);
      setContextPreview(response.contextPreview ?? null);
    } catch (err) {
      const detail = err instanceof Error ? err.message : 'Coach is unavailable right now.';
      setError(detail);
      setMessages((prev) => prev.filter((msg) => msg.id !== pendingId));
    } finally {
      setPending(false);
    }
  };

  const onKeyDown: KeyboardEventHandler<HTMLTextAreaElement> = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  const contextTitle = useMemo(() => {
    if (selectedRunLabel) return `Using ${selectedRunLabel}`;
    if (latestRunId) return `Using run ${latestRunId}`;
    return 'Pulling your latest reports + notes';
  }, [latestRunId, selectedRunLabel]);

  return (
    <Page>
      <Header>
        <Brand>
          <BrandMark src="/coach.png" alt="Coach D" />
          <BrandText>
            <BrandTitle>Coach D</BrandTitle>
            <BrandSubtitle>Form, equipment, bow tuning</BrandSubtitle>
          </BrandText>
        </Brand>
        <HeaderActions>
          <HomeButton type="button" onClick={onBack} aria-label="Back to dashboard">
            <svg viewBox="0 0 24 24" role="img" aria-hidden="true" width="20" height="20">
              <path d="M12 3.172 3.5 10.5V21h6v-5h5v5h6v-10.5L12 3.172z" />
            </svg>
          </HomeButton>
        </HeaderActions>
      </Header>

      <Layout>
        <CoachHero>
          <Glow />
          <CoachName>Talk to your coach.</CoachName>
          <CoachLine>Concise, direct answers tailored to your form, your gear, and how your bow is tuned.</CoachLine>
          <Badge>GPT-5 | Chat prompt</Badge>
          <RunSelectRow>
            <ContextLabel>Choose shot for context</ContextLabel>
            <RunSelect
              value={selectedRunId ?? ''}
              onChange={(event) => {
                const value = event.target.value || null;
                setSelectedRunId(value);
                if (!value) {
                  setSelectedRunLabel('Latest run');
                  return;
                }
                const match = availableRuns.find((item) => item.id === value);
                setSelectedRunLabel(match?.label ?? value);
              }}
              aria-label="Select run for coach chat context"
            >
              <option value="">Latest run</option>
              {availableRuns.map((run) => (
                <option key={run.id} value={run.id}>
                  {run.label || run.id}
                </option>
              ))}
            </RunSelect>
          </RunSelectRow>
        </CoachHero>

        <ChatCard>
          <ChatHeader>
            <ChatTitle>Coach chat</ChatTitle>
            <ContextHint>{contextTitle}</ContextHint>
          </ChatHeader>
          <ChatStream ref={streamRef} aria-live="polite">
            {messages.map((msg) => (
              <Bubble key={msg.id} $role={msg.role} $pending={msg.pending}>
                {msg.text}
              </Bubble>
            ))}
          </ChatStream>
          <InputArea>
            <PromptField
              placeholder="Ask about your release, timing, follow-through, or whatever feels off."
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={onKeyDown}
            />
            <SendButton type="button" onClick={handleSend} disabled={pending} $busy={pending}>
              {pending ? 'Coaching...' : 'Send to Coach'}
            </SendButton>
          </InputArea>
          {error && <ErrorText>{error}</ErrorText>}
        </ChatCard>
      </Layout>

      <ContextCard>
        <ContextTitle>Context used for responses</ContextTitle>
        <ContextRow>
          <ContextLabel>Latest run data (cleaned)</ContextLabel>
          <ToggleButton type="button" onClick={() => setShowContext((open) => !open)}>
            {showContext ? 'Hide context' : 'Show context'}
          </ToggleButton>
          {showContext && (
            <ContextBox>
              {contextPreview?.trim() || "We'll automatically inject your newest report data, personal notes, and profile."}
            </ContextBox>
          )}
        </ContextRow>
        <FooterNote>RAG, notes, and your latest report data feed every reply.</FooterNote>
      </ContextCard>
    </Page>
  );
}
