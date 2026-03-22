* API Independence:
       * Updated app/llm.py, app/stt.py, and
         app/tts.py to support OpenAI-compatible
         APIs (including Whisper and OpenAI
         TTS).
       * Added api_key support across all
         configuration sections.
       * Implemented a sounddevice fallback in
         app/pipeline.py for cross-platform
         audio recording and playback (tested
         for macOS compatibility).
   * Cooking Assistant Logic
     (run_cooking_assistant.py):
       * Proactive Vision: A background thread
         now streams the camera feed to the UI
         at 10fps, allowing Reachy to "watch"
         you cook even when not speaking.
       * Agentic Tasks: Added a regex-based
         timer system ("set a timer for 5
         minutes").
       * Placeholders: Integrated
         buy_from_instacart for ingredient
         ordering and sign_with_antennas for
         deaf-accessible feedback.
       * Cooking Persona: The system prompt is
         now tailored for professional culinary
         guidance and ingredient discovery via
         vision.
   * Modern Frontend (static/cooking.html):
       * A single-page, kitchen-themed interface
         served by default at
         http://localhost:8090.
       * Includes a live camera feed, real-time
         chat history, and a Sign Language
         Dictionary reference.
   * Accessibility:
       * Added a new perform_sign method in
         app/movements.py that uses Reachy's
         antennas to communicate with deaf
         users.

  How to Run

   1. Update your configuration: Edit
      config/settings.yaml to include your API
      keys and switch backends:
    1     llm:
    2       backend: "openai"
    3       model: "gpt-4o" # or your preferred
      VLM
    4       api_key: "your-key-here"
    5       base_url: "https://api.openai.com"
    6     stt:
    7       backend: "openai"
    8       api_key: "your-key-here"
    9     tts:
   10       backend: "openai"
   11       api_key: "your-key-here"
   12       voice: "alloy"
   2. Launch the Assistant:
   1     python3 run_cooking_assistant.py
   3. Access the UI: Open your browser to
      http://localhost:8090 to see the live feed
      and interact with Reachy.
