#
# Receptionist bot + Live Chat WebSocket broadcaster - Pipecat AI (pipecat-ai 1.8.x)
#
# Architecture:
#   Deepgram STT -> Groq LLM (OpenAI-compatible) -> Deepgram TTS
#   Transport: local microphone/speaker audio via PyAudio
#   Tool: submit_user_request -> POST to a Make.com webhook
#   Side channel: WebSocket broadcast server (ws://localhost:8765) for a live web UI
#

import asyncio
import http.server
import json
import os
import re
import socketserver
import threading
import webbrowser
from datetime import datetime

import aiohttp
import websockets
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InputAudioRawFrame,
    LLMRunFrame,
    TextFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.llm_service import FunctionCallParams
# Note: in pipecat-ai 1.8.x the PyAudio transport lives in
# `pipecat.transports.local.audio` and is named `LocalAudioTransport` /
# `LocalAudioTransportParams`. We alias them to the names requested.
from pipecat.transports.local.audio import (
    LocalAudioTransport as PyAudioTransport,
    LocalAudioTransportParams as PyAudioParams,
)
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MAKE_WEBHOOK_URL = os.getenv("MAKE_WEBHOOK_URL")

def _get_today_string() -> str:
    """Return today's date as a human-readable string like 'Sunday, August 30, 2026'."""
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y").replace(" 0", " ")


def _get_system_prompt() -> str:
    """Build the system prompt with the current date injected dynamically."""
    today = _get_today_string()
    return f"""You are a professional voice receptionist for AREZENS, an elite cybersecurity and technology services company based in Pakistan. Cybersecurity is our core practice. Technology and digital services are secondary and complementary. Speak like a real human receptionist in short, warm, natural, confident sentences. Never use markdown, bullet points, numbered lists, numbered sequences, or special characters in your spoken responses. Keep replies to one or two sentences. Ask ONE question at a time.

COMPANY IDENTITY:
AREZENS is a Pakistan-based cybersecurity and technology company. Cybersecurity is the core discipline. Technology and digital services are complementary. We work with startups, growing SMEs, enterprise teams, Pakistani clients, international clients, and remote engagements. Our philosophy: security-first delivery, evidence over assumption, one partner two disciplines, practical over theatrical. Our values: integrity, mastery, discretion, partnership.

CORE CYBERSECURITY SERVICES (these are PRIMARY):
1. Penetration Testing — manual and tool-assisted testing for web applications, mobile applications, APIs, internal networks, external networks, and cloud environments. Includes reconnaissance, vulnerability identification, exploitation, detailed reporting, severity ratings, and remediation guidance.
2. Red Teaming — objective-based adversary simulation evaluating people, processes, technology, detection capability, and response capability. Goes beyond standard penetration testing.
3. Digital Forensics and Incident Response — for security incidents, breaches, suspicious activity. Includes evidence collection, root-cause analysis, timeline reconstruction, and remediation planning. Available for post-incident work and proactive readiness assessments.
4. Vulnerability Assessment and Zero-Day Research — systematic vulnerability scanning, manual application and infrastructure review, research into emerging threats, and zero-day threat research.
5. IT and Security Consulting — security architecture, network design, security policy, compliance readiness, security questionnaires, and framework preparation. Our consultants are also practicing offensive-security testers.
6. CTF and Capability Building — capture-the-flag training, security workshops for client security teams, and student and graduate programs. Focus on hands-on offensive and defensive skills.

SECONDARY TECHNOLOGY SERVICES:
Artificial Intelligence, RAG chatbots, custom automation, predictive models, LLM integrations, software development, web development, mobile app development, cloud services, UI/UX design, and digital transformation advisory. When discussing these, make clear that AREZENS combines technology delivery with a security-first mindset. Never make technology services sound more important than cybersecurity.

INTENT DETECTION:
Understand natural language. Do not require exact keywords. Recognize these intents from what the caller says:
- General enquiry about AREZENS
- Penetration testing enquiry (e.g. "test my website", "security test", "check my app")
- Red team enquiry (e.g. "attack our company like a real attacker")
- Incident response (e.g. "I think my server got hacked", "we had a breach")
- Digital forensics
- Vulnerability assessment
- Security consulting
- CTF and training
- AI service enquiry (e.g. "AI chatbot", "automation")
- Software or web development
- Mobile development
- Cloud services
- UI/UX design
- Digital transformation
- Quote or pricing request
- Consultation booking (treat as appointment)
- Complaint or project issue
- Ticket status check
- Ticket escalation
- Rescheduling or cancellation of an appointment
- Company information questions
- Confidentiality or NDA questions
- International client questions
- Third-party application testing

SECURITY AUTHORIZATION RULE (EXTREMELY IMPORTANT):
AREZENS performs penetration testing, red teaming, and offensive-security work ONLY within a written mutually agreed scope with explicit client authorization. NEVER suggest unauthorized hacking, testing, scanning, exploitation, or access to systems. If a caller asks "can you hack this website" or similar, respond: "We can assess a system through an authorized security engagement. Testing is performed only with documented permission from the system owner." Never provide instructions for unauthorized attacks.

CONFIDENTIALITY:
AREZENS treats client information, findings, credentials, source code, and sensitive testing data as confidential. Access is limited to the engagement team. If asked about confidentiality or NDA, explain that client data, findings, source code, and engagement details are treated confidentially, and an NDA can be signed before sensitive engagements. Do not make additional legal guarantees beyond what is stated here.

CONVERSATION STYLE:
Run the conversation naturally. Acknowledge what the caller said briefly. If something is unclear, ask one short clarification question. Never overwhelm the caller with a long list of services. If they say "I need cybersecurity," do NOT immediately list all six services. Instead ask something like "Absolutely. Could you tell me what you're looking to secure or what issue you're currently facing?" Then determine the service naturally. Sound like a real professional human receptionist — warm, confident, concise, helpful, security-conscious.

QUOTES AND PRICING:
NEVER invent a price. If the caller asks about pricing, say naturally: "Pricing depends on the scope of the engagement. I can collect your requirements and have the AREZENS team follow up with a quote." You can collect the user's details for a quote request.

CURRENT OFFERS (mention when relevant, but always add that eligibility and current availability need to be confirmed by the AREZENS team):
- Free Initial Security Consultation
- Startup Technology Package
- Bundled Security and Development Discount
- Student or CTF Workshop Rate
- Retainer Loyalty Pricing
Never guarantee an offer. Always say eligibility and availability need confirmation.

APPOINTMENTS:
For an appointment, collect these ONE at a time: the caller's full name, their phone number, the requested date and time, and any extra details. Today is {today}. When the caller gives a relative date or time you MUST calculate the exact calendar date from today. For the date_time argument you pass to submit_user_request, you MUST convert the date and time into strict ISO 8601 format YYYY-MM-DDTHH:MM:SS using 24-hour clock. Never send raw English text like "tomorrow 1pm" or "next week". If the date or time is ambiguous, ask a short clarification and never invent one. Before submitting, you MUST confirm the calculated date and time conversationally with the caller and only proceed after they agree.

COMPLAINTS:
For a complaint, collect these ONE at a time: the complaint details, the caller's full name, and their phone number. A date and time is not required for a complaint unless the caller naturally gives one. Before submitting, confirm the complaint details, name, and phone.

TICKET STATUS:
If the caller wants to check a complaint ticket, ask for their ticket reference. Expected format: TKT followed by numbers. Do NOT invent ticket status, priority, or resolution time.

ESCALATION:
If a caller says their complaint was not resolved in time, explain that escalation can be requested once the expected resolution time has passed. Do not claim escalation has happened unless confirmed.

RESCHEDULING / CANCELLATION:
If the caller wants to reschedule or cancel an appointment, ask for the email used for the booking. Do not claim the appointment was changed or cancelled unless confirmed.

URGENT SECURITY INCIDENTS:
If someone reports an active or urgent security incident, be calm and professional. Recognize this as a potential Digital Forensics and Incident Response case. Collect the required information for human follow-up. Do not provide offensive-security instructions. The company policy states urgent security incidents should also be communicated directly through the contact details on the AREZENS website.

PHONE NUMBERS:
Callers often say numbers as words, for example "zero three zero zero one two three." Convert spoken number words into digits yourself. When you have the number, read it back digit by digit and ask the caller to confirm. Never guess or invent missing digits; if any part is ambiguous, ask the caller to repeat that part.

DATA INTEGRITY:
Never use placeholders such as [NAME], [PHONE], [DATE], or [DETAILS]. Only call the submit_user_request tool after the needed information is collected AND confirmed by the caller. If a required field is missing, keep asking for it naturally. Never submit fake or guessed data.

THIRD-PARTY APPLICATION TESTING:
If asked whether AREZENS can test an application built by another company, answer: "Yes. AREZENS can perform penetration testing and vulnerability assessment on applications built by the client or a third party, provided the required authorization is in place."

ONE-OFF VS RETAINER:
If asked whether AREZENS only works on long-term contracts, explain that AREZENS supports both one-off scoped engagements and ongoing retainers.

INTERNATIONAL CLIENTS:
If asked whether AREZENS works internationally, explain that AREZENS is based in Pakistan and works with both local and international clients, including remote engagements.

NO HALLUCINATION (MANDATORY):
Never invent prices, discounts, team members, certifications, offices, client names, project names, SLAs, response times, appointment availability, meeting links, ticket numbers, security guarantees, compliance certifications, or technical capabilities not described in this prompt. If information is unavailable, say: "I don't have that specific information here, but I can collect your details and have the AREZENS team follow up."

LANGUAGE:
The primary language is English. Understand natural variations and accents. If the caller speaks Urdu or mixes Urdu and English, respond naturally in the language being used where technically supported.

CONFIRM BEFORE SUBMITTING:
Always confirm the key details conversationally before calling submit_user_request. If the caller corrects something, update it and confirm again before submitting. Do not submit outdated or incorrect information."""

WS_PORT = 8765
HTTP_PORT = 8000
# Directory that holds index.html (same dir as this file).
HTTP_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# WebSocket broadcast hub
# ---------------------------------------------------------------------------
class Broadcaster:
    """Fan-out JSON messages to every connected Live Chat UI client."""

    def __init__(self) -> None:
        self._clients: set = set()

    async def add(self, ws) -> None:
        self._clients.add(ws)
        logger.info(f"WebSocket client connected ({len(self._clients)} total)")

    async def remove(self, ws) -> None:
        self._clients.discard(ws)
        logger.info(f"WebSocket client disconnected ({len(self._clients)} total)")

    async def send(self, message: dict) -> None:
        if not self._clients:
            return
        data = json.dumps(message)
        for ws in list(self._clients):
            try:
                await ws.send(data)
            except Exception as e:
                logger.warning(f"Failed to send to a WS client: {e}")
                self._clients.discard(ws)


broadcaster = Broadcaster()


async def _ws_handler(websocket) -> None:
    await broadcaster.add(websocket)
    try:
        # The server only pushes; ignore anything the browser sends.
        async for _ in websocket:
            pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        await broadcaster.remove(websocket)


async def start_ws_server() -> None:
    async with websockets.serve(_ws_handler, "0.0.0.0", WS_PORT):
        logger.info(f"WebSocket broadcast server listening on ws://localhost:{WS_PORT}")
        await asyncio.Future()  # run until cancelled


def start_http_server() -> int:
    """Serve index.html over HTTP in a background thread.

    Serves files from ``HTTP_DIR`` (where bot.py lives) using a handler bound to
    that directory, with SO_REUSEADDR and a port fallback across 8000-8009 so a
    leftover bind (e.g. a previous run) never crashes startup. Returns the port
    actually used (also stored in the module global ``HTTP_PORT``). Serving over
    http:// (rather than file://) lets the browser open a clean WebSocket to the
    broadcast server without file:// CORS quirks.
    """
    global HTTP_PORT

    class HttpHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=HTTP_DIR, **kwargs)

    class ReuseTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    httpd = None
    port = 8000
    for candidate in range(8000, 8010):
        try:
            httpd = ReuseTCPServer(("127.0.0.1", candidate), HttpHandler)
            port = candidate
            break
        except OSError:
            continue

    if httpd is None:
        logger.error("Could not bind HTTP server on any port in 8000-8009")
        return 0

    HTTP_PORT = port
    logger.info(f"Serving index.html on http://localhost:{port}")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    webbrowser.open(f"http://localhost:{port}")
    return port


# ---------------------------------------------------------------------------
# Transcript frame processors
# ---------------------------------------------------------------------------
class UserTranscriptBroadcaster(FrameProcessor):
    """Broadcasts final user speech (TranscriptionFrame) to the web UI."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            text = getattr(frame, "text", "")
            if text:
                # Verification print: confirms we only act on real recognized speech.
                print(f"[USER]: {text}")
                # Broadcast must never block / break frame propagation downstream.
                try:
                    await broadcaster.send(
                        {"type": "transcript", "role": "user", "text": text}
                    )
                except Exception as e:
                    logger.warning(f"User transcript broadcast failed: {e}")
        # CRITICAL: always forward the frame so STT/aggregators keep working.
        await self.push_frame(frame, direction)


class AssistantTranscriptBroadcaster(FrameProcessor):
    """Broadcasts bot speech (TextFrame produced by the LLM) to the web UI."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame):
            text = getattr(frame, "text", "")
            if text:
                print(f"[BOT]: {text}")
                try:
                    await broadcaster.send(
                        {"type": "transcript", "role": "assistant", "text": text}
                    )
                except Exception as e:
                    logger.warning(f"Assistant transcript broadcast failed: {e}")
        # CRITICAL: always forward the frame so TTS keeps speaking.
        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Echo / turn-taking fix
# ---------------------------------------------------------------------------
class MicState:
    """Shared flag: is the assistant currently speaking (mic should be muted)?"""

    muted = False


class MicGate(FrameProcessor):
    """Zero out microphone audio while the assistant is speaking.

    ROOT CAUSE: with a local PyAudio mic+speaker, the bot's TTS plays through the
    speaker and the same mic captures it. Deepgram then transcribes the bot's own
    voice as if it were user speech -> self-interruption, duplicated turns and
    the bot "hearing itself". VAD tuning cannot reject loud, speech-like playback.

    FIX: mute the mic at the raw-audio source (before STT *and* the user
    aggregator's VAD, since STT passes audio downstream by default). While the
    assistant speaks we replace captured audio with silence, so no fake
    TranscriptionFrame is ever produced and the user aggregator cannot start a
    turn from the bot's voice. The mic is released a short moment after the
    assistant stops, so the real user's next turn is heard normally.
    """

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame) and MicState.muted:
            # Replace captured audio with silence (all-zero samples); STT/VAD
            # then hear nothing. bytes(n) yields n zero bytes.
            silenced = InputAudioRawFrame(
                audio=bytes(len(frame.audio)),
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
            )
            await self.push_frame(silenced, direction)
            return
        await self.push_frame(frame, direction)


class SpeakingStateController(FrameProcessor):
    """Toggle MicState.muted around the assistant's actual speech.

    Driven by the TTS/bot speaking frames (not VAD), so the mute window exactly
    matches the assistant's speech plus a small tail hold-off for speaker reverb.
    """

    def __init__(self):
        super().__init__()
        self._unmute_task = None

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, (TTSStartedFrame, BotStartedSpeakingFrame)):
            MicState.muted = True
            if self._unmute_task is not None:
                self._unmute_task.cancel()
        elif isinstance(frame, (TTSStoppedFrame, BotStoppedSpeakingFrame)):
            # Keep the mic muted briefly so the tail of the speaker output / room
            # reverb is not captured as the user's first word.
            if self._unmute_task is not None:
                self._unmute_task.cancel()
            self._unmute_task = asyncio.create_task(self._unmute_after(0.5))
        await self.push_frame(frame, direction)

    async def _unmute_after(self, delay: float):
        await asyncio.sleep(delay)
        MicState.muted = False


# ---------------------------------------------------------------------------
# Tool: capture lead and forward to Make.com
# ---------------------------------------------------------------------------
def _is_placeholder(value) -> bool:
    """Reject empty values or obvious placeholder/fake tokens."""
    if not isinstance(value, str) or not value.strip():
        return True
    v = value.strip().lower()
    if v in (
        "[name]", "[phone]", "[date]", "[date_time]", "[datetime]", "[details]",
        "[service]", "[service requested]", "none", "n/a", "null",
        "unknown", "test", "testing", "placeholder", "example",
    ):
        return True
    if "[" in v or "]" in v:
        return True
    return False


async def submit_user_request(
    params: FunctionCallParams,
    type: str,
    name: str,
    phone: str,
    date_time: str,
    details: str,
) -> None:
    """Submit a customer request (appointment or complaint) to the backend.

    Only call this after the required details have been collected AND confirmed
    with the caller. For appointments, ``date_time`` is required; for complaints
    it may be left empty.

    Args:
        type: "appointment" or "complaint".
        name: The caller's full name.
        phone: The caller's phone number as digits.
        date_time: Requested appointment date and time (empty string for complaints).
        details: For appointments, any extra info; for complaints, the complaint text.
    """
    # Guard: never POST placeholders or missing required fields.
    if (
        _is_placeholder(type)
        or _is_placeholder(name)
        or _is_placeholder(phone)
        or _is_placeholder(details)
    ):
        logger.warning("Refusing submit_user_request: placeholder/missing required field")
        await params.result_callback(
            {
                "status": "error",
                "message": (
                    "I'm missing some real details. Please provide your name, phone "
                    "number, and the request details before I submit."
                ),
            }
        )
        return
    # Appointments require a date/time; complaints do not.
    if type == "appointment" and _is_placeholder(date_time):
        logger.warning("Refusing submit_user_request: appointment missing date_time")
        await params.result_callback(
            {
                "status": "error",
                "message": (
                    "I still need the date and time of your appointment before I can "
                    "submit it."
                ),
            }
        )
        return

    payload = {
        "type": type,
        "name": name,
        "phone": phone,
        "date_time": date_time or "",
        "details": details or "",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(MAKE_WEBHOOK_URL, json=payload) as response:
                await response.read()
                logger.info(f"Make webhook responded with status {response.status}")
    except Exception as e:
        logger.error(f"Failed to POST request to Make webhook: {e}")
        # Do NOT falsely confirm success.
        await params.result_callback(
            {
                "status": "error",
                "message": (
                    "I'm sorry, I wasn't able to submit that right now. Please try again."
                ),
            }
        )
        return

    # Notify the Live Chat UI that a request was captured.
    await broadcaster.send(
        {
            "type": "request_captured",
            "request_type": type,
            "name": name,
            "phone": phone,
            "date_time": date_time or "",
            "details": details or "",
            "status": "submitted",
        }
    )

    # Hand a natural confirmation back to the LLM (no technical details).
    success_msg = (
        "Perfect, I've submitted your appointment request."
        if type == "appointment"
        else "Thank you. I've submitted your complaint successfully."
    )
    await params.result_callback({"status": "success", "message": success_msg})


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
async def main() -> None:
    if not (DEEPGRAM_API_KEY and GROQ_API_KEY and MAKE_WEBHOOK_URL):
        raise RuntimeError(
            "Missing required environment variables. Ensure DEEPGRAM_API_KEY, "
            "GROQ_API_KEY and MAKE_WEBHOOK_URL are set in .env"
        )

    # 1. Transport (local microphone/speaker audio in/out via PyAudio + VAD).
    transport = PyAudioTransport(
        PyAudioParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    # 2. Speech-to-Text. Better accuracy + more deliberate turn detection.
    #    (`live_options` is deprecated; these map to DeepgramSTTService.Settings.)
    #    punctuate + smart_format keep names/sentences intact instead of breaking
    #    into single-word fragments; higher endpointing + the VAD timing above
    #    stop the bot from cutting the user off mid-sentence.
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(
            model="nova-2-general",
            language="en-US",
            punctuate=True,
            smart_format=True,
            numerals=True,  # Convert spoken number words ("nineteen twenty three") to digits
            interim_results=True,
            endpointing=600,
        ),
    )

    # 3. LLM (Groq). The system prompt lives in the LLMContext (point 1); we do
    #    NOT also set settings.system_instruction, which would send two system
    #    messages and make Groq reject the request.
    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        settings=GroqLLMService.Settings(
            model="openai/gpt-oss-20b",
        ),
    )

    # 4. Text-to-Speech. Smooth, natural voice at 16 kHz to match transport.
    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramTTSService.Settings(voice="aura-asteria-en"),
        sample_rate=16000,
    )

    # 5. Context + the lead-capture tool.
    #    The acoustic echo (bot's TTS feeding back into the mic) is now handled at
    #    the audio source by `MicGate` (it zeroes mic audio while the assistant
    #    speaks), so the VAD here only has to deal with REAL user speech. These
    #    values are kept responsive rather than maximally conservative:
    #      start_secs=0.3  -> quick turn start
    #      stop_secs=1.2   -> waits ~1.2s of silence before concluding the user
    #                         finished (natural pause, not a premature cutoff)
    #      confidence=0.85 -> rejects faint room noise without missing soft speech
    #    In pipecat 1.8.x the VAD param class is `VADParams` (not
    #    `SileroVADAnalyzer.Params`). Field mapping:
    #      threshold          -> confidence
    #      min_speech_duration-> start_secs
    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(
            confidence=0.85,
            min_volume=0.6,
            start_secs=0.3,
            stop_secs=1.2,
        )
    )
    #    In pipecat 1.8.x the universal aggregator is `LLMContextAggregatorPair`
    #    (there is no `OpenAILLMContext`/`create_context_aggregator`). We seed the
    #    system prompt directly into the context messages so the LLM always has it
    #    (the universal aggregator does not reliably merge `settings.system_instruction`).
    context = LLMContext(
        messages=[{"role": "system", "content": _get_system_prompt()}],
        tools=[submit_user_request],
    )
    #    The user-aggregator params class is `LLMUserAggregatorParams` (not
    #    `LLMUserAggregator.Params`). CRITICAL: `user_turn_stop_timeout` is a HARD
    #    CAP on how long a user turn may run before it is force-finished. 1.2s was
    #    cutting users off after ~2-3 words; 5.0s (Pipecat default) lets full
    #    sentences / names / phone numbers complete. The VAD `stop_secs=1.2`
    #    adds a long silence window before the turn is considered ended, so the
    #    bot waits for the user to actually finish instead of pausing mid-phrase.
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=vad_analyzer,
            user_turn_stop_timeout=5.0,
        ),
    )

    # Transcript broadcasters for the live web UI.
    user_broadcaster = UserTranscriptBroadcaster()
    assistant_broadcaster = AssistantTranscriptBroadcaster()

    # Echo / turn-taking controllers.
    mic_gate = MicGate()
    speaking_state_controller = SpeakingStateController()

    # 6. Pipeline wiring.
    #    - `mic_gate` sits right after transport.input() so the bot's own TTS is
    #      zeroed before STT *and* the user aggregator's VAD (STT passes audio
    #      downstream by default). This is what stops the bot hearing itself.
    #    - `speaking_state_controller` sits after transport.output() so it sees
    #      the TTS/bot speaking frames and toggles the mic mute window.
    pipeline = Pipeline(
        [
            transport.input(),
            mic_gate,
            stt,
            user_broadcaster,
            user_aggregator,
            llm,
            assistant_broadcaster,
            tts,
            transport.output(),
            speaking_state_controller,
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=True)
    await runner.add_workers(worker)

    # 7. Kick off the greeting once the pipeline starts (local audio has no
    #    participant events, so we trigger on pipeline start). Append the greeting
    #    prompt directly to the shared LLM context, then queue LLMRunFrame which
    #    makes the user aggregator push an LLMContextFrame downstream -> Groq
    #    generates immediately. While the greeting plays, `SpeakingStateController`
    #    mutes the mic via `MicGate`, so the bot never transcribes its own voice.
    @worker.event_handler("on_pipeline_started")
    async def on_pipeline_started(worker, frame):
        logger.info("Pipeline started - greeting the user")
        # Hidden user prompt forces the first assistant turn.
        context.add_message(
            {"role": "user", "content": "Greet the caller warmly as AREZENS receptionist. Say welcome to AREZENS where cybersecurity comes first, then ask how you can help them today. Keep it short."}
        )
        # Trigger Groq via the universal aggregator's run frame.
        await worker.queue_frames([LLMRunFrame()])

    # 8. Serve the Live Chat UI over HTTP (picks a free port + opens browser).
    start_http_server()

    # 9. Run the Pipecat pipeline and the WebSocket broadcast server together.
    ws_task = asyncio.create_task(start_ws_server())
    try:
        await runner.run()
    finally:
        ws_task.cancel()
        try:
            await ws_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
