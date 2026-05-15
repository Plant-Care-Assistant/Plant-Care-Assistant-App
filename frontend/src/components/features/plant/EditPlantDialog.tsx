'use client';

import { useEffect, useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { useUpdatePlantMutation } from '@/hooks/usePlants';
import { UserPlant, UserPlantUpdate } from '@/types';

interface EditPlantDialogProps {
  plant: UserPlant;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

/** Formats an ISO date for a <input type="date">; returns '' when null/undefined. */
function isoToDateInput(iso: string | null | undefined): string {
  if (!iso) return '';
  return iso.slice(0, 10); // YYYY-MM-DD
}

export const EditPlantDialog: React.FC<EditPlantDialogProps> = ({
  plant,
  open,
  onOpenChange,
}) => {
  const update = useUpdatePlantMutation();

  const [customName, setCustomName] = useState(plant.custom_name ?? '');
  const [note, setNote] = useState(plant.note ?? '');
  const [interval, setInterval] = useState<string>(
    plant.preferred_watering_interval_days?.toString() ?? '',
  );
  const [sproutedAt, setSproutedAt] = useState(isoToDateInput(plant.sprouted_at));

  // Re-seed form whenever a different plant is opened (or the same plant gets
  // refreshed mid-edit via cache invalidation).
  useEffect(() => {
    if (open) {
      setCustomName(plant.custom_name ?? '');
      setNote(plant.note ?? '');
      setInterval(plant.preferred_watering_interval_days?.toString() ?? '');
      setSproutedAt(isoToDateInput(plant.sprouted_at));
    }
  }, [open, plant]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const updates: UserPlantUpdate = {
      custom_name: customName.trim() || null,
      note: note.trim() || null,
      preferred_watering_interval_days: interval.trim()
        ? Math.max(1, parseInt(interval, 10) || 0)
        : null,
      sprouted_at: sproutedAt ? new Date(sproutedAt).toISOString() : null,
    };
    update.mutate(
      { id: plant.id, updates },
      { onSuccess: () => onOpenChange(false) },
    );
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Edit plant</DialogTitle>
          <DialogDescription>
            Update the name, watering interval, and notes for this plant.
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-1">
            <Label htmlFor="edit-name">Name</Label>
            <Input
              id="edit-name"
              value={customName}
              onChange={(e) => setCustomName(e.target.value)}
              placeholder="e.g. Living-room fern"
              maxLength={100}
            />
          </div>

          <div className="space-y-1">
            <Label htmlFor="edit-interval">Watering interval (days)</Label>
            <Input
              id="edit-interval"
              type="number"
              min={1}
              max={60}
              value={interval}
              onChange={(e) => setInterval(e.target.value)}
              placeholder="Leave blank to use the species default"
            />
            <p className="text-xs text-neutral-500">
              Overrides the species catalog default. Used for the day-cycle countdown.
            </p>
          </div>

          <div className="space-y-1">
            <Label htmlFor="edit-sprouted">Got this plant on</Label>
            <Input
              id="edit-sprouted"
              type="date"
              value={sproutedAt}
              onChange={(e) => setSproutedAt(e.target.value)}
            />
          </div>

          <div className="space-y-1">
            <Label htmlFor="edit-note">Notes</Label>
            <Textarea
              id="edit-note"
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="Anything else worth remembering — species, spot in the house, quirks…"
              rows={3}
            />
          </div>

          <DialogFooter>
            <Button
              type="button"
              variant="ghost"
              onClick={() => onOpenChange(false)}
              disabled={update.isPending}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={update.isPending}>
              {update.isPending ? 'Saving…' : 'Save changes'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};
