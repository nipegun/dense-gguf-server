/**
 * Parameter source - indicates whether a parameter uses default or custom value
 */
export enum ParameterSource {
  cCustom = 'custom',
  cDefault = 'default'
}

/**
 * Syncable parameter type - data types for parameters that can be synced with server
 */
export enum SyncableParameterType {
  cBoolean = 'boolean',
  cNumber = 'number',
  cString = 'string'
}

/**
 * Settings field type - defines the input type for settings fields
 */
export enum SettingsFieldType {
  cCheckbox = 'checkbox',
  INPUT = 'input',
  cRadio = 'radio',
  cSelect = 'select',
  cTextarea = 'textarea'
}
