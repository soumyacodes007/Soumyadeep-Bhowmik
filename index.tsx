// index.tsx
import { GoogleGenAI, Modality, Session } from '@google/genai';
import type { Blob as GenAIBlob } from '@google/genai';

const API_KEY = process.env.GEMINI_API_KEY; // Vite will replace this
const MODEL_NAME = 'gemini-2.0-flash-live-001';
const TARGET_SAMPLE_RATE = 16000; // Gemini expects 16kHz PCM
const WORKLET_BUFFER_SIZE = 4096; // How many 16kHz samples to buffer in worklet before sending
const IMAGE_SEND_INTERVAL_MS = 5000; // Send image every 5 seconds

// Helper function to encode ArrayBuffer to Base64
function arrayBufferToBase64(buffer: ArrayBuffer): string {
  let binary = '';
  const bytes = new Uint8Array(buffer);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return window.btoa(binary);
}

// Helper function to decode Base64 to ArrayBuffer
function base64ToArrayBuffer(base64: string): ArrayBuffer {
  const binaryString = window.atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
}

class GeminiLiveVoiceApp {
  private genAI: GoogleGenAI;
  private recordButton: HTMLButtonElement;
  private micIcon: HTMLElement;
  private recordText: HTMLElement;
  private recordingStatus: HTMLSpanElement;
  private recordWavesSVG: SVGSVGElement;

  private imageUploadInput: HTMLInputElement;
  private imagePreviewContainer: HTMLDivElement;
  private imagePreview: HTMLImageElement;
  private removeImageButton: HTMLButtonElement;

  private currentImageBase64: string | null = null;
  private currentImageMimeType: string | null = null;

  private session: Session | null = null;
  private isRecording: boolean = false; // For voice
  private audioContext: AudioContext | null = null;
  private micStream: MediaStream | null = null;
  private micSourceNode: MediaStreamAudioSourceNode | null = null;
  private audioWorkletNode: AudioWorkletNode | null = null;

  private lastSessionHandle: string | null = null;
  private audioQueue: ArrayBuffer[] = [];
  private isPlayingAudio: boolean = false;

  private isSetupComplete: boolean = false;
  private imageSendIntervalId: number | null = null;

  constructor() {
    if (!API_KEY) {
      this.updateStatus('API Key not found. Please set GEMINI_API_KEY.', true);
      const button = document.getElementById('recordButton') as HTMLButtonElement | null;
      if (button) button.disabled = true;
      return;
    }
    this.genAI = new GoogleGenAI({ apiKey: API_KEY, apiVersion: 'v1alpha' });

    this.recordButton = document.getElementById('recordButton') as HTMLButtonElement;
    this.micIcon = document.getElementById('micIcon') as HTMLElement;
    this.recordText = document.getElementById('recordText') as HTMLElement;
    this.recordingStatus = document.getElementById('recordingStatus') as HTMLSpanElement;
    this.recordWavesSVG = document.querySelector('.record-waves') as SVGSVGElement;

    this.imageUploadInput = document.getElementById('imageUpload') as HTMLInputElement;
    this.imagePreviewContainer = document.getElementById('imagePreviewContainer') as HTMLDivElement;
    this.imagePreview = document.getElementById('imagePreview') as HTMLImageElement;
    this.removeImageButton = document.getElementById('removeImageButton') as HTMLButtonElement;

    this.recordButton.addEventListener('click', () => this.toggleRecording());
    this.imageUploadInput.addEventListener('change', (event) => this.handleImageUpload(event));
    this.removeImageButton.addEventListener('click', () => this.removeImage());

    this.loadSessionHandle();
    this.updateStatus('Click "Talk" or Upload an Image');
  }

  private updateStatus(message: string, isError: boolean = false): void {
    this.recordingStatus.textContent = message;
    this.recordingStatus.style.color = isError ? 'var(--color-recording)' : 'var(--color-text-secondary)';
    if (isError) console.error(`[Status Error] ${message}`); else console.log(`[Status] ${message}`);
  }

  private loadSessionHandle(): void {
    this.lastSessionHandle = localStorage.getItem('geminiLiveSessionHandle');
    if (this.lastSessionHandle) {
      console.log(`[App] Found previous session handle: ${this.lastSessionHandle.substring(0, 10)}...`);
    }
  }

  private async handleImageUpload(event: Event): Promise<void> {
    const target = event.target as HTMLInputElement;
    if (target.files && target.files[0]) {
      const imageFile = target.files[0];
      const mimeType = imageFile.type;

      const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
      if (!validTypes.includes(mimeType)) {
        this.updateStatus('Invalid image type. Please use JPG, PNG, or WEBP.', true);
        this.removeImage();
        return;
      }

      const reader = new FileReader();
      reader.onload = (e) => {
        const base64Full = e.target?.result as string;
        this.currentImageBase64 = base64Full.substring(base64Full.indexOf(',') + 1);
        this.currentImageMimeType = mimeType;

        this.imagePreview.src = base64Full;
        this.imagePreviewContainer.style.display = 'block';
        this.updateStatus('Image loaded. It will be sent periodically during voice chat.');
      };
      reader.onerror = () => {
        this.updateStatus('Error reading image file.', true);
        this.removeImage();
      };
      reader.readAsDataURL(imageFile);
    }
  }

  private removeImage(): void {
    this.currentImageBase64 = null;
    this.currentImageMimeType = null;
    this.imagePreview.src = '#';
    this.imagePreviewContainer.style.display = 'none';
    this.imageUploadInput.value = '';
    this.updateStatus('Image removed. Click "Talk" or Upload an Image');
  }

  private startPeriodicImageSending(): void {
    if (this.imageSendIntervalId) {
      clearInterval(this.imageSendIntervalId);
      this.imageSendIntervalId = null;
    }
    if (!this.session || !this.isSetupComplete || !this.isRecording) { // Ensure recording is also active
        console.log("[startPeriodicImageSending] Conditions not met (session, setup, or recording). Not starting image send interval.");
        return;
    }

    console.log("[startPeriodicImageSending] Starting periodic image sending interval.");
    this.imageSendIntervalId = window.setInterval(() => {
      this.sendPeriodicImageData();
    }, IMAGE_SEND_INTERVAL_MS);

    if (this.currentImageBase64 && this.currentImageMimeType) {
        this.sendPeriodicImageData(); // Send one immediately
    }
  }

  private stopPeriodicImageSending(): void {
    if (this.imageSendIntervalId) {
      console.log("[stopPeriodicImageSending] Stopping periodic image sending interval.");
      clearInterval(this.imageSendIntervalId);
      this.imageSendIntervalId = null;
    }
  }

  private sendPeriodicImageData(): void {
    if (this.currentImageBase64 && this.currentImageMimeType && this.session && this.isRecording && this.isSetupComplete) {
      console.log(`[sendPeriodicImageData] Sending current image. MimeType: ${this.currentImageMimeType}`);
      const imageBlob: GenAIBlob = {
        data: this.currentImageBase64,
        mimeType: this.currentImageMimeType
      };
      try {
        this.session.sendRealtimeInput({ media: imageBlob });
      } catch (e) {
        console.error("[sendPeriodicImageData] Error sending image:", e);
      }
    }
  }

  private async initializeAudioSystem(): Promise<boolean> {
    console.log("[initializeAudioSystem] Attempting to initialize audio system...");
    if (!this.audioContext) {
      try {
        this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        console.log(`[initializeAudioSystem] AudioContext initialized. Native Sample Rate: ${this.audioContext.sampleRate}`);
        if (this.audioContext.state === 'suspended') {
          await this.audioContext.resume();
          console.log('[initializeAudioSystem] AudioContext resumed.');
        }
        try {
            const workletCode = `
                class AudioProcessor extends AudioWorkletProcessor {
                  constructor(options) { super(); this.sampleRate = sampleRate; this.targetSampleRate = options.processorOptions.targetSampleRate || 16000; this.bufferSize = options.processorOptions.bufferSize || 4096; const minInternalBufferSize = Math.ceil(this.bufferSize * (this.sampleRate / this.targetSampleRate)) + 128; this._internalBuffer = new Float32Array(Math.max(minInternalBufferSize, this.bufferSize * 2)); this._internalBufferIndex = 0; this.isProcessing = false; this.lastSendTime = currentTime; this.MAX_BUFFER_AGE_SECONDS = 0.5; this.resampleRatio = this.sampleRate / this.targetSampleRate; this.port.postMessage({ debug: \`Worklet Initialized. NativeSR: \${this.sampleRate}, TargetSR: \${this.targetSampleRate}, Ratio: \${this.resampleRatio}, InternalBuffer: \${this._internalBuffer.length}\` }); }
                  process(inputs, outputs, parameters) { const inputChannel = inputs[0] && inputs[0][0]; if (inputChannel && inputChannel.length > 0) { if (this._internalBufferIndex + inputChannel.length <= this._internalBuffer.length) { this._internalBuffer.set(inputChannel, this._internalBufferIndex); this._internalBufferIndex += inputChannel.length; } else { const remainingSpace = this._internalBuffer.length - this._internalBufferIndex; if (remainingSpace > 0) { this._internalBuffer.set(inputChannel.slice(0, remainingSpace), this._internalBufferIndex); this._internalBufferIndex += remainingSpace; } } } const minInputSamplesForOneOutputBuffer = Math.floor(this.bufferSize * this.resampleRatio); const shouldSendByTime = (currentTime - this.lastSendTime > this.MAX_BUFFER_AGE_SECONDS && this._internalBufferIndex > 0); const shouldSendByFill = (this._internalBufferIndex >= minInputSamplesForOneOutputBuffer); if ((shouldSendByFill || shouldSendByTime) && !this.isProcessing) { this.sendResampledBuffer(); } return true; }
                  sendResampledBuffer() { if (this._internalBufferIndex === 0 || this.isProcessing) { return; } this.isProcessing = true; this.lastSendTime = currentTime; const outputBuffer = new Float32Array(this.bufferSize); let outputIndex = 0; let consumedInputSamples = 0; for (let i = 0; i < this.bufferSize; i++) { const P = i * this.resampleRatio; const K = Math.floor(P); const T = P - K; if (K + 1 < this._internalBufferIndex) { outputBuffer[outputIndex++] = this._internalBuffer[K] * (1 - T) + this._internalBuffer[K + 1] * T; } else if (K < this._internalBufferIndex) { outputBuffer[outputIndex++] = this._internalBuffer[K]; } else { break; } consumedInputSamples = K + 1; } const finalOutputBuffer = outputBuffer.slice(0, outputIndex); if (finalOutputBuffer.length === 0) { this.port.postMessage({ debug: "Worklet sendResampledBuffer: finalOutputBuffer is empty."}); this.isProcessing = false; return; } const pcmData = new Int16Array(finalOutputBuffer.length); for (let i = 0; i < finalOutputBuffer.length; i++) { const sample = Math.max(-1, Math.min(1, finalOutputBuffer[i])); pcmData[i] = sample * 32767; } this.port.postMessage({ debug: \`Worklet posting pcmData. Output Length: \${pcmData.length}, Consumed Input Approx: \${consumedInputSamples}\` }); this.port.postMessage({ pcmData: pcmData.buffer }, [pcmData.buffer]); if (consumedInputSamples > 0 && consumedInputSamples <= this._internalBufferIndex) { this._internalBuffer.copyWithin(0, consumedInputSamples, this._internalBufferIndex); this._internalBufferIndex -= consumedInputSamples; } else { this._internalBufferIndex = 0; } this.isProcessing = false; }
                }
                registerProcessor('audio-processor', AudioProcessor);
            `;
            const blob = new Blob([workletCode], { type: 'application/javascript' });
            const workletURL = URL.createObjectURL(blob);
            await this.audioContext.audioWorklet.addModule(workletURL);
            URL.revokeObjectURL(workletURL);
        } catch (e) { console.error('[initializeAudioSystem] Failed to add AudioWorklet module:', e); this.updateStatus('Error loading audio processor.', true); return false; }
      } catch (e) { console.error('[initializeAudioSystem] Failed to create or resume AudioContext:', e); this.updateStatus('Error initializing audio system.', true); return false; }
    } else if (this.audioContext.state === 'suspended') {
        try { await this.audioContext.resume(); } catch (e) { console.error('[initializeAudioSystem] Failed to resume existing AudioContext:', e); this.updateStatus('Error resuming audio system.', true); return false; }
    }
    return true;
  }

  private async connectToGeminiIfNeeded(): Promise<boolean> {
    if (this.session && this.isSetupComplete) {
        console.log("[connectToGeminiIfNeeded] Session already exists and setup is complete.");
        return true;
    }
    // If session exists but setup not complete, we still need to wait for onmessage setupComplete
    // So, we don't re-initiate connection here, just indicate we're waiting.
    // The caller (startRecording) will check isSetupComplete again.
    if (this.session && !this.isSetupComplete) {
        console.log("[connectToGeminiIfNeeded] Session exists but setup not complete. Waiting for setup signal.");
        return true; // Indicate connection process is "active" or "pending setup"
    }
    console.log("[connectToGeminiIfNeeded] No session, (re)connecting...");
    this.isSetupComplete = false;
    return this.connectToGemini();
  }

  private async connectToGemini(): Promise<boolean> {
    this.updateStatus('Connecting to Gemini...');
    console.log("[connectToGemini] Attempting to connect...");
    this.isSetupComplete = false;
    try {
      if (this.session) {
          console.warn("[connectToGemini] Existing session found. Closing it before creating a new one.");
          try { this.session.close(); } catch (e) { console.warn("[connectToGemini] Error closing previous session:", e); }
          this.session = null;
      }
      this.session = await this.genAI.live.connect({
        model: MODEL_NAME, config: { responseModalities: [Modality.AUDIO], },
        callbacks: {
          onopen: () => { console.log("[connectToGemini] WebSocket onopen: Connection established."); this.updateStatus('Connected to Gemini! Finalizing setup...'); },
          onmessage: (eventMessage) => {
            const response = eventMessage;
            if (response?.setupComplete) {
              console.log("[connectToGemini] Received Setup complete.");
              this.isSetupComplete = true;
              this.updateStatus('Ready to talk or Upload Image');
              if (this.isRecording) { // If recording was already true (user clicked talk before setup)
                  console.log("[connectToGemini - onmessage] Setup complete and isRecording=true. Calling startRecording to finalize mic.");
                  this.startRecording(); // This will now pass the setup checks and start mic & periodic image sending
              } else if (this.currentImageBase64) { // If not recording, but an image is loaded, start periodic sending
                  this.startPeriodicImageSending(); // This will only start if isRecording becomes true later
              }
            }
            if (response?.serverContent?.modelTurn?.parts) {
              response.serverContent.modelTurn.parts.forEach(part => {
                if (part.text) { console.log(`[connectToGemini] Gemini Text (not displayed): ${part.text}`); }
                if (part.inlineData?.data && typeof part.inlineData.data === 'string') {
                  try { const audioArrayBuffer = base64ToArrayBuffer(part.inlineData.data); this.enqueueAudio(audioArrayBuffer); } catch (e) { console.error("[connectToGemini] Error decoding base64 audio from server:", e); }
                } else if (part.inlineData?.data) { console.warn("[connectToGemini] Received inlineData.data that is not a string. Type:", typeof part.inlineData.data); }
              });
            }
            if (response?.serverContent?.turnComplete) { console.log('[connectToGemini] Received Gemini turn complete.'); if (!this.isRecording && this.isSetupComplete) { this.updateStatus('Ready to talk or Upload Image'); } }
          },
          onerror: (errorEvent: ErrorEvent) => { const errorMessage = (errorEvent as any).message || (errorEvent as any).error?.message || 'Unknown WebSocket error'; console.error("[connectToGemini] WebSocket onerror triggered:", errorEvent, "Message:", errorMessage); this.updateStatus(`WebSocket Error: ${errorMessage}`, true); this.cleanupAfterErrorOrClose(true); },
          onclose: (closeEvent: CloseEvent) => { let statusMsg = 'Disconnected.'; if (!closeEvent.wasClean && this.isRecording) { statusMsg = `Disconnected unexpectedly (Code: ${closeEvent.code})`; this.updateStatus(statusMsg, true); } else if (closeEvent.code === 1000 && !this.isRecording) { statusMsg = 'Call ended.'; this.updateStatus(statusMsg); } else if (closeEvent.code !== 1000) { statusMsg = `Disconnected (Code: ${closeEvent.code})`; this.updateStatus(statusMsg); } else { this.updateStatus(statusMsg); } console.warn(`[connectToGemini] WebSocket onclose: Code ${closeEvent.code}, Reason: ${closeEvent.reason}, WasClean: ${closeEvent.wasClean}`); this.cleanupAfterErrorOrClose(false); },
        },
      });
      console.log("[connectToGemini] Connection promise resolved successfully."); return true;
    } catch (error) { console.error("[connectToGemini] Error during ai.live.connect() call:", error); this.updateStatus(`Connection setup failed: ${error instanceof Error ? error.message : String(error)}`, true); this.cleanupAfterErrorOrClose(true); return false; }
  }

  private cleanupAfterErrorOrClose(isErrorOrigin: boolean = false): void {
    console.log(`[cleanupAfterErrorOrClose] Cleaning up. Is error origin: ${isErrorOrigin}`);
    if (this.isRecording) { this.isRecording = false; }
    this.isSetupComplete = false;
    this.stopPeriodicImageSending();
    this.cleanupAudioNodes();
    if (this.session) { this.session = null; }
    this.clearAudioQueueAndStopPlayback();
    this.updateButtonUI(); 
    this.recordButton.disabled = false;
    if (!isErrorOrigin && !this.recordingStatus.textContent.toLowerCase().includes("error") && !this.recordingStatus.textContent.toLowerCase().includes("disconnect")) {
        if (this.recordingStatus.textContent !== 'Call ended.') { this.updateStatus('Ready to talk or Upload Image'); }
    }
  }

  private enqueueAudio(audioArrayBuffer: ArrayBuffer): void {
    this.audioQueue.push(audioArrayBuffer); if (!this.isPlayingAudio) { this.playNextInQueue(); }
  }

  private async playNextInQueue(): Promise<void> {
    if (this.audioQueue.length === 0) { this.isPlayingAudio = false; if (!this.isRecording && this.isSetupComplete) this.updateStatus('Ready to talk or Upload Image'); else if (this.isRecording) this.updateStatus('Listening...'); return; }
    this.isPlayingAudio = true; const audioArrayBuffer = this.audioQueue.shift()!;
    if (audioArrayBuffer.byteLength % 2 !== 0) { console.warn(`[playNextInQueue] audioArrayBuffer byteLength (${audioArrayBuffer.byteLength}) is odd.`); }
    if (audioArrayBuffer.byteLength < 2) { console.warn(`[playNextInQueue] audioArrayBuffer is too short. Skipping.`); this.isPlayingAudio = false; this.playNextInQueue(); return; }
    if (!this.audioContext || this.audioContext.state !== 'running') { const audioSystemReady = await this.initializeAudioSystem(); if (!audioSystemReady || !this.audioContext) { this.updateStatus('AudioContext not available for playback.', true); this.isPlayingAudio = false; this.audioQueue.unshift(audioArrayBuffer); return; } }
    try { const PLAYBACK_SAMPLE_RATE = 24000; const int16Array = new Int16Array(audioArrayBuffer); const float32Array = new Float32Array(int16Array.length); for (let i = 0; i < int16Array.length; i++) { float32Array[i] = int16Array[i] / 32768.0; } const audioBuffer = this.audioContext.createBuffer(1, float32Array.length, PLAYBACK_SAMPLE_RATE); audioBuffer.copyToChannel(float32Array, 0); const source = this.audioContext.createBufferSource(); source.buffer = audioBuffer; source.connect(this.audioContext.destination); source.start(); this.updateStatus('Playing Gemini response...'); source.onended = () => { this.playNextInQueue(); };
    } catch (error) { console.error("[playNextInQueue] Error playing audio:", error); this.updateStatus(`Error playing audio: ${error instanceof Error ? error.message : String(error)}`, true); this.isPlayingAudio = false; this.playNextInQueue(); }
  }

  private clearAudioQueueAndStopPlayback(): void {
    this.audioQueue = []; this.isPlayingAudio = false; console.log("[App] Audio queue cleared.");
  }

  private async toggleRecording(): Promise<void> {
    console.log(`[toggleRecording] Current state isRecording: ${this.isRecording}`);
    const audioSystemReady = await this.initializeAudioSystem(); if (!audioSystemReady) { this.updateStatus("Audio system failed to initialize.", true); return; }
    
    if (this.isRecording) { 
        this.stopRecording(); // This will set isRecording = false and stop periodic image sending
    } else { 
        this.isRecording = true; // Set intent to record
        this.updateButtonUI(); 
        await this.startRecording(); // Attempt to start recording process
    }
  }

  private async startRecording(): Promise<void> {
    console.log("[startRecording] Attempting to start recording (voice)...");
    if (!this.audioContext) { 
        this.updateStatus("Audio system not ready.", true); 
        this.isRecording = false; // Failed to start
        this.updateButtonUI(); 
        return; 
    }
    
    // Ensure connection and setup are complete before proceeding to mic.
    // connectToGeminiIfNeeded handles connecting if no session, or waiting if session exists but not setup.
    const connectedAndSetup = await this.connectToGeminiIfNeeded();
    if (!connectedAndSetup || !this.isSetupComplete) { 
        this.updateStatus(this.isSetupComplete ? "Connection failed. Cannot start recording." : "Waiting for connection setup...", !this.isSetupComplete ? false : true);
        // If connectToGeminiIfNeeded initiated a connection, onmessage for setupComplete will call startRecording again
        // if isRecording is still true. If connection failed, isRecording should be reset.
        if (!connectedAndSetup) this.isRecording = false; // Explicitly reset if connection itself failed
        this.updateButtonUI(); 
        return; 
    }

    // Re-check isRecording because async operations might have allowed user to click stop
    if (!this.isRecording) { 
        console.warn("[startRecording] isRecording became false after connection/setup checks. Aborting mic start.");
        this.updateButtonUI(); // Ensure UI is correct
        return;
    }

    this.updateStatus('Requesting microphone...');
    try {
      this.micStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 } });
      this.micSourceNode = this.audioContext.createMediaStreamSource(this.micStream);
      this.audioWorkletNode = new AudioWorkletNode(this.audioContext, 'audio-processor', { processorOptions: { targetSampleRate: TARGET_SAMPLE_RATE, bufferSize: WORKLET_BUFFER_SIZE } });
      this.audioWorkletNode.port.onmessage = (event) => {
        if (event.data.debug) { console.log(`[AudioWorklet DEBUG]: ${event.data.debug}`); return; }
        if (event.data.pcmData && this.session && this.isRecording) {
          const pcmArrayBuffer = event.data.pcmData as ArrayBuffer; if (pcmArrayBuffer.byteLength === 0) { return; }
          const base64AudioData = arrayBufferToBase64(pcmArrayBuffer);
          const audioMediaBlob: GenAIBlob = { data: base64AudioData, mimeType: `audio/pcm;rate=${TARGET_SAMPLE_RATE}` };
          if (this.session && this.isRecording) { this.session.sendRealtimeInput({ media: audioMediaBlob }); }
        }
      };
      this.micSourceNode.connect(this.audioWorkletNode);
      this.recordButton.disabled = false; 
      this.updateStatus('Listening...');
      this.startPeriodicImageSending(); // Start sending images now that voice is active
    } catch (error) { 
        console.error('[startRecording] Error in mic/worklet setup:', error); 
        this.updateStatus(`Mic/AudioWorklet error: ${error instanceof Error ? error.message : String(error)}`, true); 
        this.isRecording = false; 
        this.updateButtonUI(); 
        this.cleanupAudioNodes(); 
    }
  }
  
  private cleanupAudioNodes(): void {
    if (this.audioWorkletNode) { this.audioWorkletNode.port.onmessage = null; this.audioWorkletNode.disconnect(); this.audioWorkletNode = null; }
    if (this.micSourceNode) { this.micSourceNode.disconnect(); this.micSourceNode = null; }
    if (this.micStream) { this.micStream.getTracks().forEach(track => track.stop()); this.micStream = null; }
    console.log("[cleanupAudioNodes] Audio nodes cleaned up.");
  }

  private stopRecording(): void {
    console.log("[stopRecording] Attempting to stop recording (voice).");
    if (!this.isRecording && !this.session) { this.isRecording = false; this.updateButtonUI(); this.cleanupAudioNodes(); return; }
    
    this.isRecording = false;
    this.stopPeriodicImageSending();
    this.updateButtonUI(); 
    this.cleanupAudioNodes();

    if (this.session) { 
        this.updateStatus('Ending call...'); 
        try { this.session.close(); } 
        catch (e) { console.warn("[stopRecording] Error calling session.close():", e); this.cleanupAfterErrorOrClose(true); }
    } else { this.updateStatus('Ready to talk or Upload Image'); }
  }

  private updateButtonUI(): void {
    if (this.isRecording) { this.recordButton.classList.add('recording'); this.micIcon.classList.remove('fa-microphone'); this.micIcon.classList.add('fa-stop'); this.recordText.textContent = 'Stop'; if (this.recordWavesSVG) this.recordWavesSVG.style.display = 'block';
    } else { this.recordButton.classList.remove('recording'); this.micIcon.classList.remove('fa-stop'); this.micIcon.classList.add('fa-microphone'); this.recordText.textContent = 'Talk'; if (this.recordWavesSVG) this.recordWavesSVG.style.display = 'none'; }
    this.recordButton.disabled = false;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  new GeminiLiveVoiceApp();
});

export {};