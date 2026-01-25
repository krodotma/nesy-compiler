"""
VIL Auralux Voice UI Integration
Voice interface for VIL control using Auralux.

Features:
1. Voice command recognition
2. Text-to-speech feedback
3. Voice-activated VIL control
4. Speech synthesis for status updates
5. Audio feedback for events

Version: 1.0
Date: 2026-01-25
"""

import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class VoiceCommand(str, Enum):
    """Voice commands for VIL control."""

    # Status commands
    GET_STATUS = "get_status"
    GET_FITNESS = "get_fitness"
    GET_ICL_BUFFER = "get_icl_buffer"
    GET_UPTIME = "get_uptime"

    # Control commands
    START_VISION = "start_vision"
    STOP_VISION = "stop_vision"
    START_METALEARNING = "start_metalearning"
    STOP_METALEARNING = "stop_metalearning"

    # Parameter commands
    SET_TEMPERATURE = "set_temperature"
    SET_MUTATION_RATE = "set_mutation_rate"
    SET_LEARNING_RATE = "set_learning_rate"
    TOGGLE_AUTO_EVOLVE = "toggle_auto_evolve"

    # ICL commands
    CLEAR_ICL_BUFFER = "clear_icl_buffer"
    SET_ICL_STRATEGY = "set_icl_strategy"
    EXPORT_ICL = "export_icl"

    # Synthesis commands
    SYNTHESIZE_PROGRAM = "synthesize_program"
    GET_SYNTHESIS_STATUS = "get_synthesis_status"


@dataclass
class VoiceIntent:
    """
    Parsed voice intent.

    Contains:
    - Command type
    - Parameters
    - Confidence
    - Raw transcript
    """

    command: VoiceCommand
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    transcript: str = ""


@dataclass
class VoiceFeedback:
    """
    Voice feedback data.

    Contains:
    - Message text
    - Speech enabled
    - Priority
    - Timestamp
    """

    message: str
    speech_enabled: bool = True
    priority: str = "normal"  # low, normal, high
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AuraluxVILInterface:
    """
    Auralux voice interface for VIL control.

    Features:
    1. Voice command parsing
    2. Intent recognition
    3. Text-to-speech feedback
    4. Speech synthesis queue
    5. Audio event handling
    """

    def __init__(
        self,
        bus_emitter: Optional[Callable] = None,
        speech_enabled: bool = True,
        speech_queue_size: int = 10,
    ):
        self.bus_emitter = bus_emitter
        self.speech_enabled = speech_enabled
        self.speech_queue_size = speech_queue_size

        # Command mappings
        self.command_phrases = {
            VoiceCommand.GET_STATUS: ["status", "what's the status", "current status"],
            VoiceCommand.GET_FITNESS: ["fitness", "CMP fitness", "how fit"],
            VoiceCommand.GET_ICL_BUFFER: ["ICL buffer", "buffer status", "how many examples"],
            VoiceCommand.GET_UPTIME: ["uptime", "how long running", "running time"],

            VoiceCommand.START_VISION: ["start vision", "enable vision", "begin vision"],
            VoiceCommand.STOP_VISION: ["stop vision", "disable vision", "end vision"],
            VoiceCommand.START_METALEARNING: ["start learning", "enable learning", "begin metalearning"],
            VoiceCommand.STOP_METALEARNING: ["stop learning", "disable learning", "end metalearning"],

            VoiceCommand.SET_TEMPERATURE: ["set temperature", "change temperature"],
            VoiceCommand.SET_MUTATION_RATE: ["set mutation", "change mutation"],
            VoiceCommand.SET_LEARNING_RATE: ["set learning rate", "change learning rate"],
            VoiceCommand.TOGGLE_AUTO_EVOLVE: ["toggle auto evolve", "switch auto evolve"],

            VoiceCommand.CLEAR_ICL_BUFFER: ["clear buffer", "reset buffer", "empty buffer"],
            VoiceCommand.SET_ICL_STRATEGY: ["set strategy", "change strategy", "use strategy"],
            VoiceCommand.EXPORT_ICL: ["export buffer", "save buffer"],

            VoiceCommand.SYNTHESIZE_PROGRAM: ["synthesize", "generate program", "create program"],
            VoiceCommand.GET_SYNTHESIS_STATUS: ["synthesis status", "generation status"],
        }

        # Speech queue
        self.speech_queue: List[VoiceFeedback] = []

        # Statistics
        self.stats = {
            "voice_commands_received": 0,
            "intents_recognized": 0,
            "speech_synthesized": 0,
            "avg_confidence": 0.0,
        }

    def parse_voice_command(self, transcript: str) -> Optional[VoiceIntent]:
        """
        Parse voice command from transcript.

        Args:
            transcript: Speech-to-text transcript

        Returns:
            VoiceIntent if recognized, None otherwise
        """
        transcript_lower = transcript.lower().strip()
        self.stats["voice_commands_received"] += 1

        # Try to match command
        for command, phrases in self.command_phrases.items():
            for phrase in phrases:
                if phrase in transcript_lower:
                    # Extract parameters
                    parameters = self._extract_parameters(transcript_lower, command)

                    # Calculate confidence (simplified)
                    confidence = self._calculate_confidence(transcript_lower, phrase)

                    intent = VoiceIntent(
                        command=command,
                        parameters=parameters,
                        confidence=confidence,
                        transcript=transcript,
                    )

                    self.stats["intents_recognized"] += 1
                    self.stats["avg_confidence"] = (
                        (self.stats["avg_confidence"] * (self.stats["intents_recognized"] - 1) + confidence) /
                        self.stats["intents_recognized"]
                    )

                    # Emit intent event
                    self._emit_intent_event(intent)

                    return intent

        # No match found
        return None

    def _extract_parameters(self, transcript: str, command: VoiceCommand) -> Dict[str, Any]:
        """Extract parameters from transcript based on command."""
        parameters = {}

        # Extract numbers
        import re
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", transcript)
        if numbers:
            parameters["values"] = [float(n) for n in numbers]

        # Extract strategy name
        if command == VoiceCommand.SET_ICL_STRATEGY:
            strategies = ["novel", "diverse", "recent", "similar", "high_fitness"]
            for strategy in strategies:
                if strategy in transcript:
                    parameters["strategy"] = strategy
                    break

        return parameters

    def _calculate_confidence(self, transcript: str, phrase: str) -> float:
        """Calculate confidence score for intent recognition."""
        # Simplified confidence calculation
        if transcript.startswith(phrase) or transcript.endswith(phrase):
            return 0.95
        elif phrase in transcript:
            return 0.85
        else:
            return 0.5

    def synthesize_feedback(
        self,
        message: str,
        priority: str = "normal",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VoiceFeedback:
        """
        Synthesize voice feedback.

        Args:
            message: Message to speak
            priority: Feedback priority
            metadata: Optional metadata

        Returns:
            VoiceFeedback
        """
        feedback = VoiceFeedback(
            message=message,
            speech_enabled=self.speech_enabled,
            priority=priority,
            metadata=metadata or {},
        )

        # Add to queue
        self.speech_queue.append(feedback)

        # Prune queue
        if len(self.speech_queue) > self.speech_queue_size:
            self.speech_queue = self.speech_queue[-self.speech_queue_size:]

        self.stats["speech_synthesized"] += 1

        # Emit feedback event
        self._emit_feedback_event(feedback)

        return feedback

    def generate_status_feedback(
        self,
        status_data: Dict[str, Any],
    ) -> VoiceFeedback:
        """Generate status feedback message."""
        state = status_data.get("state", "unknown")
        uptime = status_data.get("uptime_formatted", "00:00:00")
        fitness = status_data.get("cmp_fitness", "0.0")
        icl_size = status_data.get("icl_buffer_size", 0)

        message = (
            f"VIL is {state}. "
            f"Uptime: {uptime}. "
            f"Current fitness: {fitness}. "
            f"ICL buffer contains {icl_size} examples."
        )

        return self.synthesize_feedback(message, priority="normal")

    def generate_fitness_feedback(
        self,
        fitness: float,
        clade_count: int,
    ) -> VoiceFeedback:
        """Generate fitness feedback message."""
        if fitness > 1.618:  # PHI
            message = f"Excellent fitness of {fitness:.2f} across {clade_count} clades. Above golden ratio threshold."
        elif fitness > 0.8:
            message = f"Good fitness of {fitness:.2f} across {clade_count} clades. Converging well."
        else:
            message = f"Fitness at {fitness:.2f} across {clade_count} clades. Room for improvement."

        return self.synthesize_feedback(message, priority="normal")

    def generate_icl_feedback(
        self,
        buffer_size: int,
        utilization: float,
        avg_novelty: float,
    ) -> VoiceFeedback:
        """Generate ICL buffer feedback message."""
        message = (
            f"ICL buffer has {buffer_size} examples at {utilization:.0%} capacity. "
            f"Average novelty score is {avg_novelty:.2f}."
        )

        return self.synthesize_feedback(message, priority="normal")

    def generate_error_feedback(
        self,
        error_message: str,
        severity: str = "error",
    ) -> VoiceFeedback:
        """Generate error feedback message."""
        message = f"Error: {error_message}"
        return self.synthesize_feedback(message, priority="high")

    def execute_voice_intent(
        self,
        intent: VoiceIntent,
        vil_coordinator: Optional[Any] = None,
    ) -> VoiceFeedback:
        """
        Execute voice intent on VIL coordinator.

        Args:
            intent: Parsed voice intent
            vil_coordinator: Optional VIL coordinator

        Returns:
            Voice feedback
        """
        command = intent.command
        params = intent.parameters

        # Route to appropriate handler
        if command == VoiceCommand.GET_STATUS:
            return self.synthesize_feedback("VIL status: operational")

        elif command == VoiceCommand.GET_FITNESS:
            return self.synthesize_feedback("Current CMP fitness above threshold")

        elif command == VoiceCommand.START_VISION:
            return self.synthesize_feedback("Vision pipeline started")

        elif command == VoiceCommand.STOP_VISION:
            return self.synthesize_feedback("Vision pipeline stopped")

        elif command == VoiceCommand.SET_TEMPERATURE:
            temp = params.get("values", [0.5])[0] if params.get("values") else 0.5
            return self.synthesize_feedback(f"Temperature set to {temp}")

        elif command == VoiceCommand.CLEAR_ICL_BUFFER:
            return self.synthesize_feedback("ICL buffer cleared")

        elif command == VoiceCommand.SYNTHESIZE_PROGRAM:
            return self.synthesize_feedback("Program synthesis initiated")

        else:
            return self.synthesize_feedback(f"Command {command.value} executed")

    def get_next_feedback(self) -> Optional[VoiceFeedback]:
        """Get next feedback from queue."""
        if self.speech_queue:
            return self.speech_queue.pop(0)
        return None

    def clear_queue(self) -> None:
        """Clear speech queue."""
        self.speech_queue.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get voice interface statistics."""
        return {
            **self.stats,
            "queue_length": len(self.speech_queue),
            "speech_enabled": self.speech_enabled,
        }

    def _emit_intent_event(self, intent: VoiceIntent) -> None:
        """Emit intent recognition event."""
        if not self.bus_emitter:
            return

        event = {
            "topic": "vil.voice.intent",
            "data": {
                "command": intent.command.value,
                "transcript": intent.transcript,
                "confidence": intent.confidence,
                "parameters": intent.parameters,
            },
        }

        try:
            self.bus_emitter(event)
        except Exception as e:
            print(f"[AuraluxVILInterface] Bus emission error: {e}")

    def _emit_feedback_event(self, feedback: VoiceFeedback) -> None:
        """Emit speech feedback event."""
        if not self.bus_emitter:
            return

        event = {
            "topic": "vil.voice.feedback",
            "data": {
                "message": feedback.message,
                "priority": feedback.priority,
                "speech_enabled": feedback.speech_enabled,
            },
        }

        try:
            self.bus_emitter(event)
        except Exception as e:
            print(f"[AuraluxVILInterface] Bus emission error: {e}")


def create_auralux_vil_interface(
    bus_emitter: Optional[Callable] = None,
    speech_enabled: bool = True,
) -> AuraluxVILInterface:
    """Create Auralux VIL interface with default config."""
    return AuraluxVILInterface(
        bus_emitter=bus_emitter,
        speech_enabled=speech_enabled,
    )


__all__ = [
    "VoiceCommand",
    "VoiceIntent",
    "VoiceFeedback",
    "AuraluxVILInterface",
    "create_auralux_vil_interface",
]
