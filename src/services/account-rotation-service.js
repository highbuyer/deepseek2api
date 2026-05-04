import { listAccounts, listAccountsForOwner } from "./account-service.js";

const nextAccountIndexes = new Map();

export function clearAccountIndex(apiKeyId) {
  nextAccountIndexes.delete(apiKeyId);
}

function listApiKeyAccounts(ownerId) {
  return ownerId === "admin" ? listAccounts() : listAccountsForOwner(ownerId);
}

function resolveStartIndex(accounts, preferredAccountId) {
  const preferredIndex = accounts.findIndex((account) => account.id === preferredAccountId);
  return preferredIndex === -1 ? 0 : preferredIndex;
}

export function takeRoundRobinAccount(apiKeyRecord) {
  const accounts = listApiKeyAccounts(apiKeyRecord.ownerId);
  if (!accounts.length) {
    return null;
  }

  const nextIndex = nextAccountIndexes.get(apiKeyRecord.id);
  const currentIndex = typeof nextIndex === "number"
    ? nextIndex % accounts.length
    : resolveStartIndex(accounts, apiKeyRecord.accountId);

  nextAccountIndexes.set(apiKeyRecord.id, (currentIndex + 1) % accounts.length);
  return accounts[currentIndex];
}
