// package: sky
// file: sky.proto

import * as jspb from "google-protobuf";

export class SkyEnvelope extends jspb.Message {
  getVersion(): string;
  setVersion(value: string): void;

  hasSignal(): boolean;
  clearSignal(): void;
  getSignal(): Signal | undefined;
  setSignal(value?: Signal): void;

  getPayloadCase(): SkyEnvelope.PayloadCase;
  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): SkyEnvelope.AsObject;
  static toObject(includeInstance: boolean, msg: SkyEnvelope): SkyEnvelope.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: SkyEnvelope, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): SkyEnvelope;
  static deserializeBinaryFromReader(message: SkyEnvelope, reader: jspb.BinaryReader): SkyEnvelope;
}

export namespace SkyEnvelope {
  export type AsObject = {
    version: string,
    signal?: Signal.AsObject,
  }

  export enum PayloadCase {
    PAYLOAD_NOT_SET = 0,
    SIGNAL = 2,
  }
}

export class Signal extends jspb.Message {
  hasOffer(): boolean;
  clearOffer(): void;
  getOffer(): SessionDescription | undefined;
  setOffer(value?: SessionDescription): void;

  hasAnswer(): boolean;
  clearAnswer(): void;
  getAnswer(): SessionDescription | undefined;
  setAnswer(value?: SessionDescription): void;

  hasCandidate(): boolean;
  clearCandidate(): void;
  getCandidate(): IceCandidate | undefined;
  setCandidate(value?: IceCandidate): void;

  getTypeCase(): Signal.TypeCase;
  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): Signal.AsObject;
  static toObject(includeInstance: boolean, msg: Signal): Signal.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: Signal, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): Signal;
  static deserializeBinaryFromReader(message: Signal, reader: jspb.BinaryReader): Signal;
}

export namespace Signal {
  export type AsObject = {
    offer?: SessionDescription.AsObject,
    answer?: SessionDescription.AsObject,
    candidate?: IceCandidate.AsObject,
  }

  export enum TypeCase {
    TYPE_NOT_SET = 0,
    OFFER = 1,
    ANSWER = 2,
    CANDIDATE = 3,
  }
}

export class SessionDescription extends jspb.Message {
  getSdp(): string;
  setSdp(value: string): void;

  getType(): string;
  setType(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): SessionDescription.AsObject;
  static toObject(includeInstance: boolean, msg: SessionDescription): SessionDescription.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: SessionDescription, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): SessionDescription;
  static deserializeBinaryFromReader(message: SessionDescription, reader: jspb.BinaryReader): SessionDescription;
}

export namespace SessionDescription {
  export type AsObject = {
    sdp: string,
    type: string,
  }
}

export class IceCandidate extends jspb.Message {
  getCandidate(): string;
  setCandidate(value: string): void;

  getSdpmid(): string;
  setSdpmid(value: string): void;

  getSdpmlineindex(): number;
  setSdpmlineindex(value: number): void;

  getUsernamefragment(): string;
  setUsernamefragment(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): IceCandidate.AsObject;
  static toObject(includeInstance: boolean, msg: IceCandidate): IceCandidate.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: IceCandidate, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): IceCandidate;
  static deserializeBinaryFromReader(message: IceCandidate, reader: jspb.BinaryReader): IceCandidate;
}

export namespace IceCandidate {
  export type AsObject = {
    candidate: string,
    sdpmid: string,
    sdpmlineindex: number,
    usernamefragment: string,
  }
}

