package com.rehabresearch.datacollector.data.local

import androidx.room.TypeConverter
import com.rehabresearch.datacollector.data.local.entity.BodySide
import com.rehabresearch.datacollector.data.local.entity.Difficulty
import com.rehabresearch.datacollector.data.local.entity.ExerciseType
import com.rehabresearch.datacollector.data.local.entity.Gender
import com.rehabresearch.datacollector.data.local.entity.SessionStatus
import com.rehabresearch.datacollector.data.local.entity.SurgeryType

/** Room can't persist enums natively — store them as their name string. */
class Converters {
    @TypeConverter fun genderToString(v: Gender) = v.name
    @TypeConverter fun genderFromString(v: String) = Gender.valueOf(v)

    @TypeConverter fun surgeryToString(v: SurgeryType) = v.name
    @TypeConverter fun surgeryFromString(v: String) = SurgeryType.valueOf(v)

    @TypeConverter fun sideToString(v: BodySide) = v.name
    @TypeConverter fun sideFromString(v: String) = BodySide.valueOf(v)

    @TypeConverter fun exerciseToString(v: ExerciseType) = v.name
    @TypeConverter fun exerciseFromString(v: String) = ExerciseType.valueOf(v)

    @TypeConverter fun difficultyToString(v: Difficulty) = v.name
    @TypeConverter fun difficultyFromString(v: String) = Difficulty.valueOf(v)

    @TypeConverter fun statusToString(v: SessionStatus) = v.name
    @TypeConverter fun statusFromString(v: String) = SessionStatus.valueOf(v)
}
