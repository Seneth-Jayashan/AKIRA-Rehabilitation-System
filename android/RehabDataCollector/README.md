# Rehab Data Collector (Android)

Kotlin + Jetpack Compose data-acquisition app for your rehabilitation research
pipeline: **ESP32+IMU → this app → CSV/Room → AI training**. This is deliberately
*not* the final patient-facing rehab app — it's the therapist-operated tool that
produces the labeled dataset.

**Local-only by design.** No login, no accounts, no cloud sync. Everything lives
in the on-device Room database; CSV exports are pulled off manually (USB / file
manager / `adb pull`) for the training pipeline.

## What's implemented

- **Dashboard** (start screen) — today's patient/recording counts, live BLE status card.
- **Patient management** — Room-backed CRUD, auto-generated IDs (`P001`, `P002`, ...),
  BMI computed from height/weight.
- **BLE connect screen** — scans for a device advertising the name prefix
  `ESP32-REHAB`, connects, negotiates MTU, discovers services, subscribes to
  notifications, shows RSSI/sample rate/packet counts. Connection state moves
  through `Scanning → Connecting → DiscoveringServices → EnablingNotifications →
  Connected`, and only reports `Connected` once the notification descriptor write
  actually succeeds (not just once the GATT link is up).
- **Recording flow** — `RecordingSetup` (exercise/side/difficulty/reps/week) →
  countdown (3-2-1-GO) → **Live Sensor screen** (the core screen: live Ax/Ay/Az/Gx/Gy/Gz
  readouts, two real-time MPAndroidChart line charts, rep counter, packet/frequency
  health) → **Session Summary** (pain level, correct-movement toggle, compensation,
  assistive device, notes) → CSV export.
- **History** — list of all recorded sessions with status (recording / completed / exported).
- **CSV export** — writes exactly the schema from your spec: `Timestamp,Ax,Ay,Az,Gx,Gy,Gz,
  Exercise,Side,Patient,Week,Correct`, plus a metadata sidecar CSV per session.
- **Room database** — `patients`, `sessions`, `sensor_readings` tables with foreign keys
  and cascade delete; batched inserts so a 100Hz stream doesn't overwhelm SQLite.

## What's stubbed / needs your input before shipping

1. **ESP32 firmware UUIDs** (`ble/BleConstants.kt`) — must exactly match your firmware.
2. **Packet format** (`ble/ImuPacketParser.kt`) — a fixed 28-byte little-endian
   struct: `uint32 timestamp_ms, float ax, ay, az, gx, gy, gz`. `ImuPacket` also
   has optional magnetometer/quaternion/temperature fields defined but **not yet
   wired end-to-end** (parser doesn't populate them, Room doesn't store them, CSV
   doesn't export them) — treat those as a placeholder for a future firmware
   upgrade, not working functionality.
3. **Video recording** (spec's optional "Live Video" module) — not implemented;
   would hook into CameraX alongside the recording session. Camera/mic permissions
   were removed from the manifest since this isn't wired up — add them back
   alongside the feature if you build it.

## Firmware contract (what the ESP32 must send)

Little-endian, one BLE notification per sample, ≤ negotiated MTU:

```
uint32 timestamp_ms
float32 ax, ay, az   // g
float32 gx, gy, gz   // deg/s
```

Advertise with a device name starting with `ESP32-REHAB` so the app's scan filter finds it.

## Project structure

```
app/src/main/java/com/rehabresearch/datacollector/
  ble/            BLE constants, GATT manager, packet parsing/stats
  data/local/     Room entities, DAOs, database, type converters
  data/repository/  Patient/Session repositories (the only thing ViewModels touch)
  di/             Hilt modules (Room, BleManager)
  ui/             One package per screen (dashboard, patient, ble, recording,
                  livesensor, session, history, settings) + NavHost
  utils/          CsvExporter
```

## Data flow (sensor → disk), end to end

1. `BleManager` parses each BLE notification into an `ImuPacket` and emits it on
   a `SharedFlow`.
2. `RecordingViewModel` collects that flow, buffers packets in memory, and
   flushes to Room in batches of 50 (~0.5s at 100Hz) via
   `SessionRepository.appendSensorBatch()`.
3. **The session row is written and awaited *before* the BLE collector starts** —
   `sensor_readings` has a foreign key on `sessions.sessionId`, so writing sensor
   batches before the parent session row commits risks a constraint violation.
   This is sequenced correctly as of the latest version; if you ever restructure
   `RecordingViewModel.beginRecording()`, keep the session insert and the stream
   collector in the same coroutine, in that order.
4. On "End Session", any partial batch still in memory is flushed, then the
   session row is finalized (duration, actual reps, packet/drop counts).
5. On the Session Summary screen, saving labels reads all `sensor_readings` for
   that session back out of Room and writes the CSV + metadata sidecar to
   `Android/data/com.rehabresearch.datacollector/files/exports/`, then marks the
   session `EXPORTED`.

## Building

This was authored/edited outside Android Studio in several passes (no Android
SDK available in that environment), so treat the first Gradle sync as the real
compile check. To build:

```
1. Open the RehabDataCollector/ folder in Android Studio (Koala+ recommended).
2. Let Gradle sync — it will pull the AndroidX/Compose/Hilt/Room/MPAndroidChart deps.
3. Run on a physical device — BLE doesn't work in the emulator.
```

Do a first build in Android Studio before connecting real hardware — that'll
surface any environment-specific Gradle/AGP version mismatches faster than
reading code.
