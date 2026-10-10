package com.rehabresearch.datacollector.ui.patient

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.rehabresearch.datacollector.data.local.entity.BodySide
import com.rehabresearch.datacollector.data.local.entity.Gender
import com.rehabresearch.datacollector.data.local.entity.PatientEntity
import com.rehabresearch.datacollector.data.local.entity.SurgeryType
import com.rehabresearch.datacollector.data.repository.PatientRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import java.time.LocalDate
import java.time.ZoneOffset
import javax.inject.Inject

data class NewPatientForm(
    val name: String = "",
    val age: String = "",
    val gender: Gender = Gender.MALE,
    val heightCm: String = "",
    val weightKg: String = "",
    val surgeryType: SurgeryType = SurgeryType.KNEE_REPLACEMENT,
    val surgerySide: BodySide = BodySide.LEFT,
    val surgeryDate: LocalDate = LocalDate.now(),
    val therapistName: String = "",
    val errorMessage: String? = null,
    val saved: Boolean = false
)

@HiltViewModel
class PatientViewModel @Inject constructor(
    private val repository: PatientRepository
) : ViewModel() {

    val patients: StateFlow<List<PatientEntity>> = repository.observeAll()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    private val _form = MutableStateFlow(NewPatientForm())
    val form: StateFlow<NewPatientForm> = _form

    fun updateForm(transform: (NewPatientForm) -> NewPatientForm) {
        _form.value = transform(_form.value).copy(errorMessage = null)
    }

    fun savePatient() {
        val f = _form.value
        val age = f.age.toIntOrNull()
        val height = f.heightCm.toFloatOrNull()
        val weight = f.weightKg.toFloatOrNull()

        if (f.name.isBlank()) { setError("Name is required"); return }
        if (age == null || age <= 0 || age > 120) { setError("Enter a valid age"); return }
        if (height == null || height <= 0) { setError("Enter a valid height (cm)"); return }
        if (weight == null || weight <= 0) { setError("Enter a valid weight (kg)"); return }
        if (f.therapistName.isBlank()) { setError("Therapist name is required"); return }

        viewModelScope.launch {
            val id = repository.generateNextPatientId()
            repository.addPatient(
                PatientEntity(
                    patientId = id,
                    name = f.name.trim(),
                    age = age,
                    gender = f.gender,
                    heightCm = height,
                    weightKg = weight,
                    surgeryType = f.surgeryType,
                    surgerySide = f.surgerySide,
                    surgeryDateEpochDay = f.surgeryDate.toEpochDay(),
                    physiotherapistName = f.therapistName.trim()
                )
            )
            _form.value = NewPatientForm(saved = true)
        }
    }

    private fun setError(message: String) {
        _form.value = _form.value.copy(errorMessage = message)
    }
}
