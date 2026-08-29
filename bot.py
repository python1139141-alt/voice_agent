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

SYSTEM_PROMPT = """
You are a professional voice receptionist for a business. Speak like a real human in short, warm, natural sentences. Never use markdown, bullet points, numbered lists, or special characters.

You handle two kinds of requests: appointments and complaints. Understand the caller's intent from what they say and do not make them state the exact word.

Run the conversation naturally. Ask for only one piece of information per turn and wait for the reply before continuing. Do not repeat a question the caller already answered. Acknowledge what the caller said briefly, and if something is unclear ask one short clarification question. Keep your replies to one or two sentences.

For an appointment, collect these one at a time: the caller's full name, their phone number, the requested date and time, and any extra details. Today is Saturday, August 29, 2026. When the caller gives a relative date or time you MUST calculate the exact calendar date from today's date: for example, today is Saturday August 29 2026, so tomorrow is Sunday August 30 2026, Monday this week is Monday August 31 2026, and next Monday is Monday September 7 2026. For the date_time argument that you pass to submit_user_request, you MUST convert the date and time into strict ISO 8601 format YYYY-MM-DDTHH:MM:SS using 24 hour clock. For example, if today is August 29 2026 and the caller says tomorrow at 1 PM, date_time must be exactly 2026-08-30T13:00:00. Never send raw English text such as tomorrow 1pm, Monday at 2pm, or next week. If the date or time is ambiguous, ask a short clarification and never invent one. Before submitting, you MUST confirm the calculated date and time conversationally with the caller, for example say so that's tomorrow, Sunday, August 30 at 1 PM, right and only proceed after they agree.

For a complaint, collect these one at a time: the complaint details, the caller's full name, and their phone number. A date and time is not required for a complaint unless the caller naturally gives one, so do not force it. Before submitting, confirm the complaint details, name, and phone.

Phone numbers: callers often say numbers as words, for example zero three zero zero one two three. Convert spoken number words into digits yourself. When you have the number, read it back digit by digit and ask the caller to confirm. Never guess or invent missing digits; if any part is ambiguous, ask the caller to repeat that part.

Data integrity: never use placeholders such as [NAME], [PHONE], [DATE], or [DETAILS]. Only call the submit_user_request tool after the needed information is collected and confirmed by the caller. If a required field is missing, keep asking for it naturally. Never submit fake or guessed data.

Always confirm the key details conversationally before calling submit_user_request. If the caller corrects something, update it and confirm again before submitting, and do not submit outdated or incorrect information.
"""

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
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
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
            {"role": "user", "content": "Greet the caller warmly and ask how you can help them today."}
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
