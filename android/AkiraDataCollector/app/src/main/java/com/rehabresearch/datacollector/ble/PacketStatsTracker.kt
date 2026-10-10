package com.rehabresearch.datacollector.ble

class PacketStatsTracker(
    private val targetHz: Int = 100
) {

    private var lastTimestamp: Long? = null

    private var received = 0L

    private var dropped = 0L

    private var lastWindow = System.currentTimeMillis()

    private var windowPackets = 0

    private var sampleRate = 0f

    private val expectedInterval
        get() = 1000.0 / targetHz

    fun onPacket(packet: ImuPacket): BleLinkStats {

        received++
        windowPackets++

        lastTimestamp?.let {

            val gap = packet.timestampMillis - it

            if (gap > expectedInterval * 1.5) {

                val missed =
                    (gap / expectedInterval).toLong() - 1

                if (missed > 0)
                    dropped += missed
            }
        }

        lastTimestamp = packet.timestampMillis

        val now = System.currentTimeMillis()

        val elapsed = now - lastWindow

        if (elapsed >= 1000) {

            sampleRate = windowPackets * 1000f / elapsed

            windowPackets = 0

            lastWindow = now
        }

        return BleLinkStats(
            packetsReceived = received,
            packetsDropped = dropped,
            currentSampleRateHz = sampleRate
        )
    }

    fun reset() {

        lastTimestamp = null

        received = 0

        dropped = 0

        sampleRate = 0f

        windowPackets = 0

        lastWindow = System.currentTimeMillis()
    }
}