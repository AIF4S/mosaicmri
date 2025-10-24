<?xml version="1.0" encoding="ISO-8859-1"?>

<xsl:stylesheet version="1.0"
                xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

    <xsl:output method="xml" indent="yes"/>

    <!-- Helpers with graceful fallback -->
    <xsl:variable name="phaseOversampling">
        <xsl:choose>
            <xsl:when test="siemens/IRIS/DERIVED/phaseOversampling">
                <xsl:choose>
                    <xsl:when test="string(number(siemens/IRIS/DERIVED/phaseOversampling)) = 'NaN'">0</xsl:when>
                    <xsl:otherwise>
                        <xsl:value-of select="siemens/IRIS/DERIVED/phaseOversampling"/>
                    </xsl:otherwise>
                </xsl:choose>
            </xsl:when>
            <xsl:otherwise>0</xsl:otherwise>
        </xsl:choose>
    </xsl:variable>

    <xsl:variable name="sliceOversampling">
        <xsl:choose>
            <xsl:when test="siemens/MEAS/sKSpace/dSliceOversamplingForDialog">
                <xsl:choose>
                    <xsl:when test="string(number(siemens/MEAS/sKSpace/dSliceOversamplingForDialog)) = 'NaN'">0</xsl:when>
                    <xsl:otherwise>
                        <xsl:value-of select="siemens/MEAS/sKSpace/dSliceOversamplingForDialog"/>
                    </xsl:otherwise>
                </xsl:choose>
            </xsl:when>
            <xsl:otherwise>0</xsl:otherwise>
        </xsl:choose>
    </xsl:variable>

    <xsl:variable name="partialFourierPhase">
        <xsl:choose>
            <xsl:when test="siemens/MEAS/sKSpace/ucPhasePartialFourier = 1">0.5</xsl:when>
            <xsl:when test="siemens/MEAS/sKSpace/ucPhasePartialFourier = 2">0.625</xsl:when>
            <xsl:when test="siemens/MEAS/sKSpace/ucPhasePartialFourier = 4">0.75</xsl:when>
            <!-- DICOM.flReadoutOSFactor -->
            <xsl:if test="siemens/DICOM/flReadoutOSFactor">
              <userParameterString>
                <name>DICOM.flReadoutOSFactor</name>
                <value><xsl:value-of select="siemens/DICOM/flReadoutOSFactor"/></value>
              </userParameterString>
            </xsl:if>

              <!-- Additional DICOM XML params -->
              <xsl:if test="siemens/DICOM/dSliceResolution">
                <userParameterString>
                  <name>DICOM.dSliceResolution</name>
                  <value><xsl:value-of select="siemens.DICOM.dSliceResolution"/></value>
                </userParameterString>
              </xsl:if>
              <xsl:if test="siemens/DICOM/flUsedPatientWeight">
                <userParameterString>
                  <name>DICOM.flUsedPatientWeight</name>
                  <value><xsl:value-of select="siemens.DICOM.flUsedPatientWeight"/></value>
                </userParameterString>
              </xsl:if>
              <xsl:if test="siemens/DICOM/flTransRefAmpl">
                <userParameterString>
                  <name>DICOM.flTransRefAmpl</name>
                  <value><xsl:value-of select="siemens.DICOM.flTransRefAmpl"/></value>
                </userParameterString>
              </xsl:if>
              <xsl:if test="siemens/DICOM/flPatientAge">
                <userParameterString>
                  <name>DICOM.flPatientAge</name>
                  <value><xsl:value-of select="siemens.DICOM.flPatientAge"/></value>
                </userParameterString>
              </xsl:if>
              <xsl:if test="siemens/DICOM/flMagneticFieldStrength">
                <userParameterString>
                  <name>DICOM.flMagneticFieldStrength</name>
                  <value><xsl:value-of select="siemens.DICOM.flMagneticFieldStrength"/></value>
                </userParameterString>
              </xsl:if>
              <xsl:if test="siemens/DICOM/flPhaseOS">
                <userParameterString>
                  <name>DICOM.flPhaseOS</name>
                  <value><xsl:value-of select="siemens.DICOM.flPhaseOS"/></value>
                </userParameterString>
              </xsl:if>
              <xsl:if test="siemens/DICOM/flSliceOS">
                <userParameterString>
                  <name>DICOM.flSliceOS</name>
                  <value><xsl:value-of select="siemens.DICOM.flSliceOS"/></value>
                </userParameterString>
              </xsl:if>
            <xsl:when test="siemens/MEAS/sKSpace/ucPhasePartialFourier = 8">0.875</xsl:when>
            <xsl:otherwise>1.0</xsl:otherwise>
        </xsl:choose>
    </xsl:variable>

    <xsl:variable name="numberOfContrasts">
        <xsl:value-of select="siemens/MEAS/lContrasts"/>
    </xsl:variable>

    <!-- Patient/study id fallbacks: prefer IRIS.RECOMPOSE, else Config.* if mapped -->
  <xsl:variable name="studyID">
    <xsl:choose>
      <xsl:when test="normalize-space(siemens/IRIS/RECOMPOSE/StudyLOID)!=''">
        <xsl:value-of select="substring(siemens/IRIS/RECOMPOSE/StudyLOID, 6)"/>
      </xsl:when>
      <xsl:when test="normalize-space(siemens/HEADER/StudyLOID)!=''">
        <xsl:choose>
          <xsl:when test="starts-with(normalize-space(siemens/HEADER/StudyLOID), 'LOID:') and string-length(normalize-space(siemens/HEADER/StudyLOID)) &gt; 5">
            <xsl:value-of select="substring(normalize-space(siemens/HEADER/StudyLOID), 6)"/>
          </xsl:when>
          <xsl:otherwise>
            <xsl:value-of select="normalize-space(siemens/HEADER/StudyLOID)"/>
          </xsl:otherwise>
        </xsl:choose>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="'UNKNOWN_STUDY'"/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:variable>

  <xsl:variable name="patientID">
    <xsl:choose>
      <xsl:when test="normalize-space(siemens/IRIS/RECOMPOSE/PatientLOID)!=''">
        <xsl:value-of select="substring(siemens/IRIS/RECOMPOSE/PatientLOID, 6)"/>
      </xsl:when>
      <xsl:when test="normalize-space(siemens/IRIS/RECOMPOSE/PatientID)!=''">
        <xsl:value-of select="normalize-space(siemens/IRIS/RECOMPOSE/PatientID)"/>
      </xsl:when>
      <xsl:when test="normalize-space(siemens/HEADER/PatientLOID)!=''">
        <xsl:choose>
          <xsl:when test="starts-with(normalize-space(siemens/HEADER/PatientLOID), 'LOID:') and string-length(normalize-space(siemens/HEADER/PatientLOID)) &gt; 5">
            <xsl:value-of select="substring(normalize-space(siemens/HEADER/PatientLOID), 6)"/>
          </xsl:when>
          <xsl:otherwise>
            <xsl:value-of select="normalize-space(siemens/HEADER/PatientLOID)"/>
          </xsl:otherwise>
        </xsl:choose>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="'UNKNOWN_PATIENT'"/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:variable>

  <!-- Raw LOIDs from HEADER with safe trimming (do not fallback to other sources) -->
  <xsl:variable name="headerPatientLOID">
    <xsl:choose>
      <xsl:when test="normalize-space(siemens/HEADER/PatientLOID)!=''">
        <xsl:choose>
          <xsl:when test="starts-with(normalize-space(siemens/HEADER/PatientLOID), 'LOID:') and string-length(normalize-space(siemens/HEADER/PatientLOID)) &gt; 5">
            <xsl:value-of select="substring(normalize-space(siemens/HEADER/PatientLOID), 6)"/>
          </xsl:when>
          <xsl:otherwise>
            <xsl:value-of select="normalize-space(siemens/HEADER/PatientLOID)"/>
          </xsl:otherwise>
        </xsl:choose>
      </xsl:when>
      <xsl:otherwise></xsl:otherwise>
    </xsl:choose>
  </xsl:variable>

  <xsl:variable name="headerStudyLOID">
    <xsl:choose>
      <xsl:when test="normalize-space(siemens/HEADER/StudyLOID)!=''">
        <xsl:choose>
          <xsl:when test="starts-with(normalize-space(siemens/HEADER/StudyLOID), 'LOID:') and string-length(normalize-space(siemens/HEADER/StudyLOID)) &gt; 5">
            <xsl:value-of select="substring(normalize-space(siemens/HEADER/StudyLOID), 6)"/>
          </xsl:when>
          <xsl:otherwise>
            <xsl:value-of select="normalize-space(siemens/HEADER/StudyLOID)"/>
          </xsl:otherwise>
        </xsl:choose>
      </xsl:when>
      <xsl:otherwise></xsl:otherwise>
    </xsl:choose>
  </xsl:variable>

    <xsl:variable name="strSeperator">_</xsl:variable>

  <xsl:variable name="pixelSpacing">
    <xsl:choose>
      <xsl:when test="(siemens/MEAS/sSliceArray/asSlice/s0/dReadoutFOV &gt; 0) and (siemens/YAPS/flReadoutOSFactor &gt; 0) and (siemens/MEAS/sKSpace/lBaseResolution &gt; 0)">
        <xsl:value-of select="siemens/MEAS/sSliceArray/asSlice/s0/dReadoutFOV * 0.5 * siemens/YAPS/flReadoutOSFactor div siemens/MEAS/sKSpace/lBaseResolution"/>
      </xsl:when>
      <xsl:otherwise>1.0</xsl:otherwise>
    </xsl:choose>
  </xsl:variable>

    <xsl:template match="/">
        <ismrmrdHeader xsi:schemaLocation="http://www.ismrm.org/ISMRMRD ismrmrd.xsd"
                       xmlns="http://www.ismrm.org/ISMRMRD"
                       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                       xmlns:xs="http://www.w3.org/2001/XMLSchema">

          <subjectInformation>
              <!-- Map LOID into the standard patientID field -->
              <xsl:if test="string-length(normalize-space($headerPatientLOID)) &gt; 0 or string-length(normalize-space($patientID)) &gt; 0">
                  <patientID>
                      <xsl:choose>
                        <xsl:when test="string-length(normalize-space($headerPatientLOID)) &gt; 0">
                          <xsl:value-of select="normalize-space($headerPatientLOID)"/>
                        </xsl:when>
                        <xsl:otherwise>
                          <xsl:value-of select="normalize-space($patientID)"/>
                        </xsl:otherwise>
                      </xsl:choose>
                  </patientID>
              </xsl:if>
        <xsl:if test="normalize-space(siemens/DICOM/tPatientName)!=''">
          <patientName>
            <xsl:value-of select="normalize-space(siemens/DICOM/tPatientName)"/>
          </patientName>
        </xsl:if>
              <xsl:if test="siemens/YAPS/flUsedPatientWeight &gt; 0">
                  <patientWeight_kg>
                      <xsl:value-of select="siemens/YAPS/flUsedPatientWeight"/>
                  </patientWeight_kg>
              </xsl:if>

              <xsl:if test="siemens/YAPS/flPatientHeight &gt; 0">
                  <patientHeight_m>
                      <xsl:value-of select="siemens/YAPS/flPatientHeight"/>
                  </patientHeight_m>
              </xsl:if>

              <patientGender>
                  <xsl:choose>
                      <xsl:when test="siemens/DICOM/lPatientSex = 1">F</xsl:when>
                      <xsl:when test="siemens/DICOM/lPatientSex = 2">M</xsl:when>
                      <xsl:otherwise>O</xsl:otherwise>
                  </xsl:choose>
              </patientGender>
          </subjectInformation>

          <!-- Provide study LOID via standard studyID field under studyInformation -->
          <studyInformation>
            <xsl:if test="string-length(normalize-space($headerStudyLOID)) &gt; 0 or string-length(normalize-space($studyID)) &gt; 0">
              <studyID>
                <xsl:choose>
                  <xsl:when test="string-length(normalize-space($headerStudyLOID)) &gt; 0">
                    <xsl:value-of select="normalize-space($headerStudyLOID)"/>
                  </xsl:when>
                  <xsl:otherwise>
                    <xsl:value-of select="normalize-space($studyID)"/>
                  </xsl:otherwise>
                </xsl:choose>
              </studyID>
            </xsl:if>
          </studyInformation>

          <measurementInformation>
                <measurementID>
                    <xsl:value-of select="concat(string(siemens/DICOM/DeviceSerialNumber), $strSeperator, $patientID, $strSeperator, $studyID, $strSeperator, string(siemens/HEADER/MeasUID))"/>
                </measurementID>
        <patientPosition>
          <xsl:choose>
            <xsl:when test="normalize-space(siemens/YAPS/tPatientPosition)!=''">
              <xsl:value-of select="normalize-space(siemens/YAPS/tPatientPosition)"/>
            </xsl:when>
            <xsl:when test="normalize-space(siemens/DICOM/tPatientPosition)!=''">
              <xsl:value-of select="normalize-space(siemens/DICOM/tPatientPosition)"/>
            </xsl:when>
            <xsl:otherwise>HFS</xsl:otherwise>
          </xsl:choose>
        </patientPosition>
                <protocolName>
                    <xsl:value-of select="siemens/MEAS/tProtocolName"/>
                </protocolName>

                <xsl:if test="siemens/YAPS/ReconMeasDependencies/RFMap &gt; 0">
                    <measurementDependency>
                        <dependencyType>RFMap</dependencyType>
                        <measurementID>
                            <xsl:value-of select="concat(string(siemens/DICOM/DeviceSerialNumber), $strSeperator, $patientID, $strSeperator, $studyID, $strSeperator, string(siemens/YAPS/ReconMeasDependencies/RFMap))"/>
                        </measurementID>
                    </measurementDependency>
                </xsl:if>

                <xsl:if test="siemens/YAPS/ReconMeasDependencies/SenMap &gt; 0">
                    <measurementDependency>
                        <dependencyType>SenMap</dependencyType>
                        <measurementID>
                            <xsl:value-of select="concat(string(siemens/DICOM/DeviceSerialNumber), $strSeperator, $patientID, $strSeperator, $studyID, $strSeperator, string(siemens/YAPS/ReconMeasDependencies/SenMap))"/>
                        </measurementID>
                    </measurementDependency>
                </xsl:if>

                <xsl:if test="siemens/YAPS/ReconMeasDependencies/Noise &gt; 0">
                    <measurementDependency>
                        <dependencyType>Noise</dependencyType>
                        <measurementID>
                            <xsl:value-of select="concat(string(siemens/DICOM/DeviceSerialNumber), $strSeperator, $patientID, $strSeperator, $studyID, $strSeperator, string(siemens/YAPS/ReconMeasDependencies/Noise))"/>
                        </measurementID>
                    </measurementDependency>
                </xsl:if>

                <frameOfReferenceUID>
                    <xsl:value-of select="siemens/YAPS/tFrameOfReference" />
                </frameOfReferenceUID>

            </measurementInformation>

            <acquisitionSystemInformation>
                <systemVendor>
                    <xsl:value-of select="siemens/DICOM/Manufacturer"/>
                </systemVendor>
                <systemModel>
                    <xsl:value-of select="siemens/DICOM/ManufacturersModelName"/>
                </systemModel>
        <systemFieldStrength_T>
          <xsl:choose>
            <xsl:when test="siemens/YAPS/flMagneticFieldStrength &gt; 0">
              <xsl:value-of select="siemens/YAPS/flMagneticFieldStrength"/>
            </xsl:when>
            <xsl:otherwise>0.0</xsl:otherwise>
          </xsl:choose>
        </systemFieldStrength_T>
                <relativeReceiverNoiseBandwidth>0.793</relativeReceiverNoiseBandwidth>
        <receiverChannels>
          <xsl:choose>
            <xsl:when test="siemens/YAPS/iMaxNoOfRxChannels &gt; 0">
              <xsl:value-of select="siemens/YAPS/iMaxNoOfRxChannels" />
            </xsl:when>
            <xsl:otherwise>1</xsl:otherwise>
          </xsl:choose>
        </receiverChannels>
                <institutionName>
                    <xsl:value-of select="siemens/DICOM/InstitutionName" />
                </institutionName>
            </acquisitionSystemInformation>

          <experimentalConditions>
            <H1resonanceFrequency_Hz>
              <xsl:choose>
                <xsl:when test="siemens/DICOM/lFrequency &gt; 0">
                  <xsl:value-of select="siemens/DICOM/lFrequency"/>
                </xsl:when>
                <xsl:otherwise>0</xsl:otherwise>
              </xsl:choose>
            </H1resonanceFrequency_Hz>
          </experimentalConditions>

          <encoding>
            <trajectory>
              <xsl:choose>
                <xsl:when test="siemens/MEAS/sKSpace/ucTrajectory = 1">cartesian</xsl:when>
                <xsl:when test="siemens/MEAS/sKSpace/ucTrajectory = 2">radial</xsl:when>
                <xsl:when test="siemens/MEAS/sKSpace/ucTrajectory = 4">spiral</xsl:when>
                <xsl:when test="siemens/MEAS/sKSpace/ucTrajectory = 8">propellor</xsl:when>
                <xsl:otherwise>other</xsl:otherwise>
              </xsl:choose>
            </trajectory>

            <encodedSpace>
              <matrixSize>
                <xsl:choose>
                  <xsl:when test="siemens/MEAS/sKSpace/ucTrajectory = 1">
                    <x>
                        <xsl:choose>
                          <xsl:when test="siemens/YAPS/iNoOfFourierColumns &gt; 0">
                            <xsl:value-of select="siemens/YAPS/iNoOfFourierColumns"/>
                          </xsl:when>
                          <xsl:otherwise>
                            <xsl:value-of select="siemens/MEAS/sKSpace/lBaseResolution"/>
                          </xsl:otherwise>
                        </xsl:choose>
                    </x>
                  </xsl:when>
                  <xsl:otherwise>
                    <xsl:choose>
                      <xsl:when test="(siemens/IRIS/DERIVED/imageColumns) and (siemens/IRIS/DERIVED/imageColumns &gt; 0)">
                        <x>
                          <xsl:value-of select="siemens/IRIS/DERIVED/imageColumns"/>
                        </x>
                      </xsl:when>
                      <xsl:otherwise>
                        <x>
                          <xsl:choose>
                            <xsl:when test="siemens/MEAS/sKSpace/lBaseResolution &gt; 0">
                              <xsl:value-of select="siemens/MEAS/sKSpace/lBaseResolution"/>
                            </xsl:when>
                            <xsl:otherwise>1</xsl:otherwise>
                          </xsl:choose>
                        </x>
                      </xsl:otherwise>
                    </xsl:choose>
                  </xsl:otherwise>
                </xsl:choose>

                <xsl:choose>
                  <xsl:when test="siemens/MEAS/sKSpace/uc2DInterpolation" >
                    <xsl:choose>
                      <xsl:when test="siemens/MEAS/sKSpace/uc2DInterpolation = 1">
                        <y>
                          <xsl:value-of select="floor(siemens/YAPS/iPEFTLength div 2)"/>
                        </y>
                      </xsl:when>
                      <xsl:otherwise>
                        <y>
                          <xsl:choose>
                            <xsl:when test="siemens/YAPS/iPEFTLength &gt; 0">
                              <xsl:value-of select="siemens/YAPS/iPEFTLength"/>
                            </xsl:when>
                            <xsl:otherwise>1</xsl:otherwise>
                          </xsl:choose>
                        </y>
                      </xsl:otherwise>
                    </xsl:choose>
                  </xsl:when>
                  <xsl:otherwise>
                    <y>
                      <xsl:choose>
                        <xsl:when test="siemens/YAPS/iPEFTLength &gt; 0">
                          <xsl:value-of select="siemens/YAPS/iPEFTLength"/>
                        </xsl:when>
                        <xsl:otherwise>1</xsl:otherwise>
                      </xsl:choose>
                    </y>
                  </xsl:otherwise>
                </xsl:choose>

                <xsl:choose>
                  <xsl:when test="not(siemens/YAPS/iNoOfFourierPartitions) or (siemens/YAPS/i3DFTLength = 1)">
                    <z>1</z>
                  </xsl:when>
                  <xsl:otherwise>
                    <z>
                      <xsl:choose>
                        <xsl:when test="siemens/YAPS/i3DFTLength &gt; 0">
                          <xsl:value-of select="siemens/YAPS/i3DFTLength"/>
                        </xsl:when>
                        <xsl:otherwise>1</xsl:otherwise>
                      </xsl:choose>
                    </z>
                  </xsl:otherwise>
                </xsl:choose>
              </matrixSize>

              <fieldOfView_mm>
                <xsl:choose>
                  <xsl:when test="siemens/MEAS/sKSpace/ucTrajectory = 1">
                    <x>
                      <xsl:value-of select="siemens/MEAS/sSliceArray/asSlice/s0/dReadoutFOV * siemens/YAPS/flReadoutOSFactor"/>
                    </x>
                  </xsl:when>
                  <xsl:otherwise>
                    <x>
                      <xsl:value-of select="siemens/MEAS/sSliceArray/asSlice/s0/dReadoutFOV"/>
                    </x>
                  </xsl:otherwise>
                </xsl:choose>
                <y>
                  <xsl:value-of select="siemens/MEAS/sSliceArray/asSlice/s0/dPhaseFOV * (1+$phaseOversampling)"/>
                </y>
                <z>
                  <xsl:value-of select="siemens/MEAS/sSliceArray/asSlice/s0/dThickness * (1+$sliceOversampling)"/>
                </z>
              </fieldOfView_mm>
            </encodedSpace>

            <reconSpace>
              <matrixSize>
                <xsl:choose>
                  <xsl:when test="(siemens/IRIS/DERIVED/imageColumns) and (siemens/IRIS/DERIVED/imageColumns &gt; 0)">
                    <x>
                      <xsl:value-of select="siemens/IRIS/DERIVED/imageColumns"/>
                    </x>
                  </xsl:when>
                  <xsl:otherwise>
                    <x>
                      <xsl:choose>
                        <xsl:when test="siemens/MEAS/sKSpace/lBaseResolution &gt; 0">
                          <xsl:value-of select="siemens/MEAS/sKSpace/lBaseResolution"/>
                        </xsl:when>
                        <xsl:otherwise>1</xsl:otherwise>
                      </xsl:choose>
                    </x>
                  </xsl:otherwise>
                </xsl:choose>

                <xsl:choose>
                  <xsl:when test="(siemens/IRIS/DERIVED/imageLines) and (siemens/IRIS/DERIVED/imageLines &gt; 0)">
                    <y>
                      <xsl:value-of select="siemens/IRIS/DERIVED/imageLines"/>
                    </y>
                  </xsl:when>
                  <xsl:otherwise>
                    <y>
                      <xsl:choose>
                        <xsl:when test="(siemens/MEAS/sSliceArray/asSlice/s0/dPhaseFOV &gt; 0) and ($pixelSpacing &gt; 0)">
                          <xsl:value-of select="floor(siemens/MEAS/sSliceArray/asSlice/s0/dPhaseFOV * (1+$phaseOversampling) div $pixelSpacing + 0.5)"/>
                        </xsl:when>
                        <xsl:otherwise>1</xsl:otherwise>
                      </xsl:choose>
                    </y>
                  </xsl:otherwise>
                </xsl:choose>

                <xsl:choose>
                  <xsl:when test="siemens/YAPS/i3DFTLength = 1">
                    <z>1</z>
                  </xsl:when>
                  <xsl:otherwise>
                    <z>
                      <xsl:choose>
                        <xsl:when test="siemens/MEAS/sKSpace/lImagesPerSlab &gt; 0">
                          <xsl:value-of select="siemens/MEAS/sKSpace/lImagesPerSlab"/>
                        </xsl:when>
                        <xsl:otherwise>1</xsl:otherwise>
                      </xsl:choose>
                    </z>
                  </xsl:otherwise>
                </xsl:choose>
              </matrixSize>
              <fieldOfView_mm>
                <x>
                  <xsl:choose>
                    <xsl:when test="siemens/MEAS/sSliceArray/asSlice/s0/dReadoutFOV &gt; 0">
                      <xsl:value-of select="siemens/MEAS/sSliceArray/asSlice/s0/dReadoutFOV"/>
                    </xsl:when>
                    <xsl:otherwise>1.0</xsl:otherwise>
                  </xsl:choose>
                </x>
                <y>
                  <xsl:choose>
                    <xsl:when test="siemens/MEAS/sSliceArray/asSlice/s0/dPhaseFOV &gt; 0">
                      <xsl:value-of select="siemens/MEAS/sSliceArray/asSlice/s0/dPhaseFOV"/>
                    </xsl:when>
                    <xsl:otherwise>1.0</xsl:otherwise>
                  </xsl:choose>
                </y>
                <z>
                  <xsl:choose>
                    <xsl:when test="siemens/MEAS/sSliceArray/asSlice/s0/dThickness &gt; 0">
                      <xsl:value-of select="siemens/MEAS/sSliceArray/asSlice/s0/dThickness"/>
                    </xsl:when>
                    <xsl:otherwise>1.0</xsl:otherwise>
                  </xsl:choose>
                </z>
              </fieldOfView_mm>
            </reconSpace>

            <encodingLimits>
              <kspace_encoding_step_1>
                <minimum>0</minimum>
                <maximum>
                  <xsl:choose>
                    <xsl:when test="siemens/YAPS/iNoOfFourierLines &gt; 0">
                      <xsl:value-of select="siemens/YAPS/iNoOfFourierLines - 1"/>
                    </xsl:when>
                    <xsl:otherwise>0</xsl:otherwise>
                  </xsl:choose>
                </maximum>
                <center>
                  <xsl:choose>
                    <xsl:when test="siemens/MEAS/sKSpace/lPhaseEncodingLines &gt; 0">
                      <xsl:value-of select="floor(siemens/MEAS/sKSpace/lPhaseEncodingLines div 2)"/>
                    </xsl:when>
                    <xsl:otherwise>0</xsl:otherwise>
                  </xsl:choose>
                </center>
              </kspace_encoding_step_1>
              <kspace_encoding_step_2>
                <minimum>0</minimum>
                <xsl:choose>
                  <xsl:when test="not(siemens/YAPS/iNoOfFourierPartitions) or (siemens/YAPS/i3DFTLength = 1)">
                    <maximum>0</maximum>
                    <center>0</center>
                  </xsl:when>
                  <xsl:otherwise>
                    <maximum>
                      <xsl:choose>
                        <xsl:when test="siemens/YAPS/iNoOfFourierPartitions &gt; 0">
                          <xsl:value-of select="siemens/YAPS/iNoOfFourierPartitions - 1"/>
                        </xsl:when>
                        <xsl:otherwise>0</xsl:otherwise>
                      </xsl:choose>
                    </maximum>
                    <center>0</center>
                  </xsl:otherwise>
                </xsl:choose>
              </kspace_encoding_step_2>
              <slice>
                <minimum>0</minimum>
                <maximum>
                  <xsl:value-of select="siemens/MEAS/sSliceArray/lSize - 1"/>
                </maximum>
                <center>0</center>
              </slice>
              <set>
                <minimum>0</minimum>
                <maximum>
                  <xsl:choose>
                    <xsl:when test="siemens/YAPS/iNSet">
                      <xsl:value-of select="siemens/YAPS/iNSet - 1"/>
                    </xsl:when>
                    <xsl:otherwise>0</xsl:otherwise>
                  </xsl:choose>
                </maximum>
                <center>0</center>
              </set>
              <phase>
                <minimum>0</minimum>
                <maximum>
                  <xsl:choose>
                    <xsl:when test="siemens/MEAS/sPhysioImaging/lPhases">
                      <xsl:value-of select="siemens/MEAS/sPhysioImaging/lPhases - 1"/>
                    </xsl:when>
                    <xsl:otherwise>0</xsl:otherwise>
                  </xsl:choose>
                </maximum>
                <center>0</center>
              </phase>
              <repetition>
                <minimum>0</minimum>
                <maximum>
                  <xsl:choose>
                    <xsl:when test="siemens/MEAS/lRepetitions">
                      <xsl:value-of select="siemens/MEAS/lRepetitions"/>
                    </xsl:when>
                    <xsl:otherwise>0</xsl:otherwise>
                  </xsl:choose>
                </maximum>
                <center>0</center>
              </repetition>
              <segment>
                <minimum>0</minimum>
                <maximum>0</maximum>
                <center>0</center>
              </segment>
              <contrast>
                <minimum>0</minimum>
                <maximum>
                  <xsl:choose>
                    <xsl:when test="siemens/MEAS/lContrasts">
                      <xsl:value-of select="siemens/MEAS/lContrasts - 1"/>
                    </xsl:when>
                    <xsl:otherwise>0</xsl:otherwise>
                  </xsl:choose>
                </maximum>
                <center>0</center>
              </contrast>
              <average>
                <minimum>0</minimum>
                <maximum>
                  <xsl:choose>
                    <xsl:when test="siemens/MEAS/lAverages">
                      <xsl:value-of select="siemens/MEAS/lAverages - 1"/>
                    </xsl:when>
                    <xsl:otherwise>0</xsl:otherwise>
                  </xsl:choose>
                </maximum>
                <center>0</center>
              </average>
            </encodingLimits>

            <parallelImaging>
              <accelerationFactor>
                <kspace_encoding_step_1>
                  <xsl:choose>
                    <xsl:when test="not(siemens/MEAS/sPat/lAccelFactPE)">1</xsl:when>
                    <xsl:otherwise>
                      <xsl:value-of select="(siemens/MEAS/sPat/lAccelFactPE)"/>
                    </xsl:otherwise>
                  </xsl:choose>
                </kspace_encoding_step_1>
                <kspace_encoding_step_2>
                  <xsl:choose>
                    <xsl:when test="not(siemens/MEAS/sPat/lAccelFact3D)">1</xsl:when>
                    <xsl:otherwise>
                      <xsl:value-of select="(siemens/MEAS/sPat/lAccelFact3D)"/>
                    </xsl:otherwise>
                  </xsl:choose>
                </kspace_encoding_step_2>
              </accelerationFactor>
              <calibrationMode>
                <xsl:choose>
                  <xsl:when test="siemens/MEAS/sPat/ucRefScanMode = 1">other</xsl:when>
                  <xsl:when test="siemens/MEAS/sPat/ucRefScanMode = 2">embedded</xsl:when>
                  <xsl:when test="siemens/MEAS/sPat/ucRefScanMode = 4">separate</xsl:when>
                  <xsl:when test="siemens/MEAS/sPat/ucRefScanMode = 8">separate</xsl:when>
                  <xsl:when test="siemens/MEAS/sPat/ucRefScanMode = 16">interleaved</xsl:when>
                  <xsl:when test="siemens/MEAS/sPat/ucRefScanMode = 32">interleaved</xsl:when>
                  <xsl:when test="siemens/MEAS/sPat/ucRefScanMode = 64">interleaved</xsl:when>
                  <xsl:otherwise>other</xsl:otherwise>
                </xsl:choose>
              </calibrationMode>
            </parallelImaging>
          </encoding>

          <sequenceParameters>
            <xsl:for-each select="siemens/MEAS/alTR">
              <xsl:if test="position() = 1">
                <TR>
                  <xsl:value-of select=". div 1000.0" />
                </TR>
              </xsl:if>
              <xsl:if test="(position() &gt; 1) and (. &gt; 0)">
                <TR>
                  <xsl:value-of select=". div 1000.0" />
                </TR>
              </xsl:if>
            </xsl:for-each>
            <xsl:for-each select="siemens/MEAS/alTE">
              <xsl:if test="position() = 1">
                <TE>
                  <xsl:value-of select=". div 1000.0" />
                </TE>
              </xsl:if>
              <xsl:if test="(position() &gt; 1) and (. &gt; 0)">
                <xsl:if test="position() &lt; ($numberOfContrasts + 1)">
                  <TE>
                    <xsl:value-of select=". div 1000.0" />
                  </TE>
                </xsl:if>
              </xsl:if>
            </xsl:for-each>
            <xsl:for-each select="siemens/MEAS/alTI">
              <xsl:if test=". &gt; 0">
                <TI>
                  <xsl:value-of select=". div 1000.0" />
                </TI>
              </xsl:if>
            </xsl:for-each>
            <xsl:for-each select="siemens/DICOM/adFlipAngleDegree">
              <xsl:if test=". &gt; 0">
                <flipAngle_deg>
                  <xsl:value-of select="." />
                </flipAngle_deg>
              </xsl:if>
            </xsl:for-each>
            <xsl:if test="siemens/YAPS/lEchoSpacing">
              <echo_spacing>
                <xsl:value-of select="siemens/YAPS/lEchoSpacing div 1000.0" />
              </echo_spacing>
            </xsl:if>
          </sequenceParameters>

          <userParameters>
            <!-- Explicit LOIDs from HEADER to accompany patient info -->
            <xsl:if test="string-length(normalize-space($headerPatientLOID)) &gt; 0">
              <userParameterString>
                <name>PatientLOID</name>
                <value><xsl:value-of select="$headerPatientLOID"/></value>
              </userParameterString>
            </xsl:if>
            <xsl:if test="string-length(normalize-space($headerStudyLOID)) &gt; 0">
              <userParameterString>
                <name>StudyLOID</name>
                <value><xsl:value-of select="$headerStudyLOID"/></value>
              </userParameterString>
            </xsl:if>

            <!-- Plane rotation if present -->
            <userParameterString>
              <name>dInPlaneRot</name>
              <value>
                <xsl:choose>
                  <xsl:when test="siemens/MEAS/sSliceArray/asSlice/s0/dInPlaneRot">
                    <xsl:value-of select="siemens/MEAS/sSliceArray/asSlice/s0/dInPlaneRot"/>
                  </xsl:when>
                  <xsl:otherwise>0.0</xsl:otherwise>
                </xsl:choose>
              </value>
            </userParameterString>

            <!-- Added parameters -->
            <!-- tStudyDescription: prefer HEADER.tStudyDescription; fallback to MEAS.tProtocolName -->
            <xsl:choose>
              <xsl:when test="normalize-space(siemens/HEADER/tStudyDescription)!=''">
                <userParameterString>
                  <name>tStudyDescription</name>
                  <value><xsl:value-of select="normalize-space(siemens/HEADER/tStudyDescription)"/></value>
                </userParameterString>
              </xsl:when>
              <xsl:when test="normalize-space(siemens/MEAS/tProtocolName)!=''">
                <userParameterString>
                  <name>tStudyDescription</name>
                  <value><xsl:value-of select="normalize-space(siemens/MEAS/tProtocolName)"/></value>
                </userParameterString>
              </xsl:when>
            </xsl:choose>

            <!-- phaseOversampling from IRIS.DERIVED (already normalized in helper) -->
            <userParameterString>
              <name>phaseOversampling</name>
              <value><xsl:value-of select="$phaseOversampling"/></value>
            </userParameterString>

            <!-- ReadoutOversamplingFactor from YAPS.flReadoutOSFactor -->
            <userParameterString>
              <name>ReadoutOversamplingFactor</name>
              <value>
                <xsl:choose>
                  <xsl:when test="siemens/YAPS/flReadoutOSFactor &gt; 0">
                    <xsl:value-of select="siemens/YAPS/flReadoutOSFactor"/>
                  </xsl:when>
                  <xsl:otherwise>1.0</xsl:otherwise>
                </xsl:choose>
              </value>
            </userParameterString>

            <!-- PhaseFoV from MEAS.sSliceArray.asSlice.s0.dPhaseFOV -->
            <xsl:if test="siemens/MEAS/sSliceArray/asSlice/s0/dPhaseFOV &gt; 0">
              <userParameterString>
                <name>PhaseFoV</name>
                <value><xsl:value-of select="siemens/MEAS/sSliceArray/asSlice/s0/dPhaseFOV"/></value>
              </userParameterString>
            </xsl:if>

            <!-- BaseResolution from MEAS.sKSpace.lBaseResolution -->
            <xsl:if test="siemens/MEAS/sKSpace/lBaseResolution &gt; 0">
              <userParameterString>
                <name>BaseResolution</name>
                <value><xsl:value-of select="siemens/MEAS/sKSpace/lBaseResolution"/></value>
              </userParameterString>
            </xsl:if>

              <!-- YAPS.iNoOfFourierColumns -->
              <xsl:if test="siemens/YAPS/iNoOfFourierColumns &gt; 0">
                <userParameterString>
                  <name>YAPS.iNoOfFourierColumns</name>
                  <value><xsl:value-of select="siemens/YAPS/iNoOfFourierColumns"/></value>
                </userParameterString>
              </xsl:if>

              <!-- YAPS.iNoOfFourierLines -->
              <xsl:if test="siemens/YAPS/iNoOfFourierLines &gt; 0">
                <userParameterString>
                  <name>YAPS.iNoOfFourierLines</name>
                  <value><xsl:value-of select="siemens/YAPS/iNoOfFourierLines"/></value>
                </userParameterString>
              </xsl:if>

            <!-- NumberOfProcessors: try HEADER first, then YAPS or MEAS if present -->
            <xsl:choose>
              <xsl:when test="normalize-space(siemens/HEADER/NumberOfProcessors)!=''">
                <userParameterString>
                  <name>NumberOfProcessors</name>
                  <value><xsl:value-of select="normalize-space(siemens/HEADER/NumberOfProcessors)"/></value>
                </userParameterString>
              </xsl:when>
              <xsl:when test="normalize-space(siemens/YAPS/NumberOfProcessors)!=''">
                <userParameterString>
                  <name>NumberOfProcessors</name>
                  <value><xsl:value-of select="normalize-space(siemens/YAPS/NumberOfProcessors)"/></value>
                </userParameterString>
              </xsl:when>
              <xsl:when test="normalize-space(siemens/MEAS/NumberOfProcessors)!=''">
                <userParameterString>
                  <name>NumberOfProcessors</name>
                  <value><xsl:value-of select="normalize-space(siemens/MEAS/NumberOfProcessors)"/></value>
                </userParameterString>
              </xsl:when>
            </xsl:choose>
          </userParameters>

        </ismrmrdHeader>
      </xsl:template>

    </xsl:stylesheet>
